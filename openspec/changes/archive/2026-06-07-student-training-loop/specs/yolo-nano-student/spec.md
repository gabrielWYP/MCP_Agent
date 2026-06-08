# Delta for yolo-nano-student

## Overview

Adds freeze_backbone and unfreeze_backbone_stages method stubs to StudentModel for API compatibility with Trainer's two-phase logic. Both methods are no-ops since student training is single-phase.

## ADDED Requirements

### R1: Freeze Backbone Stub

StudentModel MUST implement `freeze_backbone(freeze_stages: int = 0)` as a no-op method. The method SHALL accept the same signature as MasterModel's freeze_backbone for API compatibility but perform no parameter freezing.

#### Scenario: Freeze backbone called during student training

- GIVEN StudentModel instance with model_type="student"
- WHEN freeze_backbone(freeze_stages=3) is called
- THEN no parameters SHALL have their requires_grad changed
- AND the method SHALL return None without error

#### Scenario: Freeze backbone with default argument

- GIVEN StudentModel instance
- WHEN freeze_backbone() is called (no arguments)
- THEN the method SHALL complete without error and change no parameters

### R2: Unfreeze Backbone Stages Stub

StudentModel MUST implement `unfreeze_backbone_stages(unfreeze_stages: list[int])` as a no-op method. The method SHALL accept the same signature as MasterModel's unfreeze_backbone_stages for API compatibility but perform no parameter modification.

#### Scenario: Unfreeze backbone stages called during student training

- GIVEN StudentModel instance with model_type="student"
- WHEN unfreeze_backbone_stages(unfreeze_stages=[3, 4]) is called
- THEN no parameters SHALL have their requires_grad changed
- AND the method SHALL return None without error

#### Scenario: Unfreeze with empty list

- GIVEN StudentModel instance
- WHEN unfreeze_backbone_stages(unfreeze_stages=[]) is called
- THEN the method SHALL complete without error and change no parameters

### R3: Method Documentation

Both freeze/unfreeze stub methods MUST include docstrings that clearly state they are no-ops for single-phase student training. Docstrings SHOULD explain the API compatibility rationale.

#### Scenario: Docstring indicates no-op behavior

- GIVEN StudentModel.freeze_backbone method
- WHEN help(StudentModel.freeze_backbone) is called or __doc__ is accessed
- THEN the docstring SHALL contain "no-op" and "single-phase"
- AND SHALL explain the method exists for API compatibility with Trainer

## Dependencies

- training-loop: Trainer calls freeze_backbone/unfreeze_backbone_stages; stubs ensure compatibility

## Non-functional

- No-op methods MUST have zero measurable overhead (<1us call time)
- Method signatures MUST match MasterModel equivalents exactly to enable polymorphic dispatch