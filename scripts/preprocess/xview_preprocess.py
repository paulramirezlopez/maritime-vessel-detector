#!/usr/bin/env python3
"""
Tile large xView maritime vessel images and convert annotations to CVAT XML.
- Keeps only type_id 23-32 (all collapsed to "ship" label)
- Tiles: 1024x1024 px, stride 824, min 50% object overlap
- Clips polygons to tile boundaries, outputs as PNG tiles + CVAT XML
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from xml.dom import minidom

# Maritime vessel type IDs to keep
MARITIME_IDS = {40, 41, 42, 44, 45, 47, 49, 50, 51, 52}
SHIP_LABEL = "ship"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile xView images and convert maritime annotations to CVAT XML."
    )
    parser.add_argument(
        "--geojson", required=True, help="Path to the original xView GeoJSON file."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing the manually selected subset of images.",
    )
    parser.add_argument(
        "--output_images_dir",
        required=True,
        help="Directory to store the tiled PNG images.",
    )
    parser.add_argument(
        "--output_xml", required=True, help="Output CVAT XML file path."
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile width and height (default 1024)",
    )
    parser.add_argument(
        "--stride", type=int, default=824, help="Stride between tiles (default 824)"
    )
    parser.add_argument(
        "--min_overlap",
        type=float,
        default=0.5,
        help="Minimum object overlap ratio (default 0.5)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip re-creating tiles that already exist",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )
    return parser.parse_args()


def clip_polygon_to_tile(poly, tile_bbox):
    """Clip a shapely Polygon to a tile rectangle. Returns the clipped Polygon or None."""
    from shapely.geometry import GeometryCollection, MultiPolygon, box

    tile_box = box(*tile_bbox)
    inter = poly.intersection(tile_box)

    if inter.is_empty:
        return None

    if isinstance(inter, MultiPolygon):
        return max(inter.geoms, key=lambda g: g.area)

    if inter.geom_type == "Polygon":
        return inter

    if isinstance(inter, GeometryCollection):
        polys = [g for g in inter.geoms if g.geom_type == "Polygon"]
        if polys:
            return max(polys, key=lambda p: p.area)

    return None


def polygon_to_cvat_points(shapely_poly):
    """Convert a shapely Polygon to a CVAT points string rounded to 2 decimal places."""
    exterior = list(shapely_poly.exterior.coords)
    if len(exterior) > 1 and exterior[0] == exterior[-1]:
        exterior = exterior[:-1]
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in exterior)


def process_image(args, img_path, features, start_tile_id):
    """
    Tile a single image and return a list of tile records for CVAT XML.
    Returns: (list_of_tiles, next_available_tile_id)
    """
    from PIL import Image
    from shapely.geometry import Polygon, box

    img = Image.open(img_path)
    width, height = img.size

    if width < args.tile_size or height < args.tile_size:
        if args.verbose:
            print(f"  {img_path.name}: smaller than tile size, skipped.")
        return [], start_tile_id

    poly_info = []
    for feat in features:
        props = feat.get("properties", {})
        coords_str = props.get("bounds_imcoords")

        if not coords_str:
            continue

        try:
            xmin, ymin, xmax, ymax = map(float, coords_str.split(","))
            poly = box(xmin, ymin, xmax, ymax)
            if poly.is_valid and poly.area > 0:
                poly_info.append(poly)
        except Exception as e:
            if args.verbose:
                print(f"    Invalid bounds_imcoords '{coords_str}': {e}")

    if args.verbose:
        print(f"  {img_path.name}: {len(poly_info)} vessel polygons found")

    tiles = []
    rows = (height - args.tile_size) // args.stride + 1
    cols = (width - args.tile_size) // args.stride + 1

    current_id = start_tile_id

    for row in range(rows):
        for col in range(cols):
            x_start = col * args.stride
            y_start = row * args.stride
            tile_bbox = (
                x_start,
                y_start,
                x_start + args.tile_size,
                y_start + args.tile_size,
            )

            tile_file = f"{img_path.stem}_row{row}_col{col}.png"
            tile_path = Path(args.output_images_dir) / tile_file

            if not args.skip_existing or not tile_path.is_file():
                img.crop(tile_bbox).save(tile_path)

            tile_annotations = []
            for poly in poly_info:
                pbox = poly.bounds
                if (
                    pbox[2] < tile_bbox[0]
                    or pbox[0] > tile_bbox[2]
                    or pbox[3] < tile_bbox[1]
                    or pbox[1] > tile_bbox[3]
                ):
                    continue

                clipped = clip_polygon_to_tile(poly, tile_bbox)
                if clipped is None:
                    continue

                try:
                    overlap = clipped.area / poly.area
                except ZeroDivisionError:
                    continue
                if overlap < args.min_overlap:
                    continue

                local_coords = [
                    (x - x_start, y - y_start) for x, y in clipped.exterior.coords
                ]
                if len(local_coords) < 3:
                    continue

                local_poly = Polygon(local_coords)
                if not local_poly.is_valid or local_poly.area <= 0:
                    continue

                tile_annotations.append(local_poly)

            tiles.append(
                {
                    "id": current_id,
                    "name": tile_file,
                    "width": args.tile_size,
                    "height": args.tile_size,
                    "annotations": tile_annotations,
                }
            )
            current_id += 1

    return tiles, current_id


def main():
    args = parse_args()

    print("Loading GeoJSON...")
    with open(args.geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        print("No features found. Exiting.")
        return

    annotations_by_image = defaultdict(list)
    for feat in features:
        props = feat.get("properties", {})
        try:
            type_id = int(props.get("type_id", -1))
        except (TypeError, ValueError):
            continue

        if type_id in MARITIME_IDS:
            img_id = str(props.get("image_id", ""))
            annotations_by_image[img_id].append(feat)

    print(
        f"Total maritime annotations mapped from GeoJSON: {sum(len(v) for v in annotations_by_image.values())}"
    )

    images_dir = Path(args.images_dir)
    valid_images = [
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"}
    ]

    print(f"Found {len(valid_images)} target images in {images_dir}")

    output_images_dir = Path(args.output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = []
    global_tile_id = 0
    total_annotations = 0

    for img_path in valid_images:
        print(f"Processing {img_path.name}...")

        img_features = annotations_by_image.get(img_path.name, [])
        if not img_features:
            img_features = annotations_by_image.get(img_path.stem, [])

        tiles, global_tile_id = process_image(
            args, img_path, img_features, global_tile_id
        )

        all_tiles.extend(tiles)
        ann_count = sum(len(t["annotations"]) for t in tiles)
        if args.verbose:
            print(f"  Created {len(tiles)} tiles with {ann_count} valid annotations")

        total_annotations += ann_count

    print(f"\nTotal tiles created: {len(all_tiles)}")
    print(f"Total annotations converted: {total_annotations}")

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "1"
    ET.SubElement(task, "name").text = "Maritime Vessels - Ship"
    ET.SubElement(task, "size").text = str(len(all_tiles))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "flipped").text = "False"
    ET.SubElement(task, "created").text = ""
    ET.SubElement(task, "updated").text = ""

    labels_elem = ET.SubElement(task, "labels")
    label = ET.SubElement(labels_elem, "label")
    ET.SubElement(label, "name").text = SHIP_LABEL
    ET.SubElement(label, "attributes")

    for tile_info in all_tiles:
        img_elem = ET.SubElement(
            root,
            "image",
            id=str(tile_info["id"]),
            name=tile_info["name"],
            width=str(tile_info["width"]),
            height=str(tile_info["height"]),
        )

        for poly in tile_info["annotations"]:
            points = polygon_to_cvat_points(poly)
            ET.SubElement(
                img_elem,
                "polygon",
                label=SHIP_LABEL,
                occluded="0",
                source="manual",
                points=points,
            )

    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

    output_xml_path = Path(args.output_xml)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"CVAT XML successfully saved to {output_xml_path}")


if __name__ == "__main__":
    main()
