"""Tests for NIR Label Studio to RGB YOLO conversion helpers."""

from scripts.convert_nir_labels import (
    bbox_center_inside,
    parse_mango_bbox_from_yolo_line,
    yolo_cxcywh_to_xyxy,
)


def test_yolo_cxcywh_to_xyxy_converts_normalized_bbox() -> None:
    assert yolo_cxcywh_to_xyxy(0.5, 0.5, 0.25, 0.5, 800, 600) == (
        300,
        150,
        500,
        450,
    )


def test_parse_mango_bbox_from_yolo_line_only_accepts_class_zero() -> None:
    assert parse_mango_bbox_from_yolo_line("1 0.5 0.5 0.2 0.2", 100, 100) is None
    assert parse_mango_bbox_from_yolo_line("0 0.5 0.5 0.2 0.2", 100, 100) == (
        40,
        40,
        60,
        60,
    )


def test_bbox_center_inside_filters_damage_outside_mango() -> None:
    mango_bbox = (100, 100, 300, 300)
    inside_damage = (150, 150, 180, 180)
    outside_damage = (310, 150, 350, 180)

    assert bbox_center_inside(inside_damage, mango_bbox)
    assert not bbox_center_inside(outside_damage, mango_bbox)
    assert bbox_center_inside(outside_damage, mango_bbox, margin=40)
