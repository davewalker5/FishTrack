# Maldives – Example Video Clips

## Overview

This directory contains a selection of underwater video clips recorded in the Maldives, provided as example footage for the _FishTrack_ project.

The clips feature marine fish observed in their natural environment and are intended to support the development, testing and demonstration of FishTrack's fish detection and tracking capabilities.

The footage includes examples of different species, swimming behaviours, environmental conditions and potential challenges for automated detection and tracking.

## Video Catalogue

The following table describes the clips included in this directory.

| Filename                                                 | Species (Common Name) | Scientific Name         | ROI Detection Method | Tracking Method |
| -------------------------------------------------------- | --------------------- | ----------------------- | -------------------- | --------------- |
| `OC-V-26-007-stabilised-trimmed.mp4`                     | Titan Triggerfish     | Balistoides viridescens | N/A                  | N/A             |
| `OC-V-26-007-stabilised-trimmed-Auto-CV-tracked.mp4`     | Titan Triggerfish     | Balistoides viridescens | Grounding DINO       | Conventional CV |
| `OC-V-26-007-stabilised-trimmed-Manual-CV-tracked.mp4`   | Titan Triggerfish     | Balistoides viridescens | Manual               | Conventional CV |
| `OC-V-26-007-stabilised-trimmed-Auto-SAM2-tracked.mp4`   | Titan Triggerfish     | Balistoides viridescens | Grounding DINO       | SAM 2           |
| `OC-V-26-007-stabilised-trimmed-Manual-SAM2-tracked.mp4` | Titan Triggerfish     | Balistoides viridescens | Manual               | SAM 2           |

**Notes:**

- **ROI Detection Method:** The method used to identify the initial region of interest containing the fish
- **Tracking Method:** The method used to track the identified fish through successive video frames
- Where multiple species appear, the species column identifies the principal subject of interest

## Purpose and Usage

These clips may be used to:

- Demonstrate FishTrack's detection and tracking capabilities
- Evaluate different ROI detection and tracking approaches
- Test performance under varying underwater conditions, including changes in lighting, visibility, background complexity and camera movement
- Provide reproducible examples for development and regression testing

The footage is provided as illustrative test material. Inclusion of a clip does not necessarily imply that FishTrack can successfully detect or track every fish appearing in it.

## Copyright and Licensing

**Copyright © 2026 David Walker.**

All video footage contained in this directory is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.

Licence details: https://creativecommons.org/licenses/by/4.0/

Under this licence, you are free to:

- **Share:** Copy and redistribute the footage in any medium or format
- **Adapt:** Remix, transform and build upon the footage for any purpose, including commercial use

Provided that you:

- Give appropriate credit to the original creator
- Provide a link to the CC BY 4.0 licence
- Indicate whether changes have been made
- Do not imply that the original creator endorses your use

### Suggested Attribution

> Maldives underwater footage © 2026 David Walker. Licensed under CC BY 4.0. https://creativecommons.org/licenses/by/4.0/

If the footage has been modified, the attribution should also describe the changes made.

### Relationship to the FishTrack Licence

The FishTrack application's source code is distributed under the **MIT Licence**, as described in the repository's root `LICENSE` file.

The video footage in this directory is a separate creative work and is **not covered by the MIT Licence**.

The CC BY 4.0 licence applies specifically to the video footage contained in this directory. It does not alter the licensing terms of the FishTrack application or its source code.

## Provenance

The original footage was recorded by David Walker in the Maldives. Clips may have been extracted, shortened, resized or re-encoded from the original recordings for use with FishTrack. Such processing does not change the ownership or licensing of the footage.
