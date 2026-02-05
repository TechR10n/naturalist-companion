# Wireframes (MVP)

These are low-fidelity wireframes meant to align the MVP user flows before building iOS UI.

## 1) Home

```text
+------------------------------------------------+
| Agentic Naturalist                              |
| "What are we seeing today?"                    |
|                                                |
| [ Route Guide ]  [ Camera ]  [ Explore ]        |
|                                                |
| Recent                                           |
| - I-81 Shenandoah (last run)                    |
| - Camera: "Layered roadcut"                    |
+------------------------------------------------+
```

## 2) Route Guide Flow

### 2.1 Pick Route

```text
+------------------------------------------------+
| Route Guide                                     |
|                                                |
| Route source                                    |
| ( ) Choose corridor                             |
| ( ) Upload GPX                                  |
| ( ) Paste points (JSON)                         |
|                                                |
| Corridor: [ I-81 Shenandoah Valley      v ]      |
| Stops (max): [ 8 ]   Spacing (km): [ 15 ]       |
| Radius (km): [ 15 ]  Language: [ en ]           |
|                                                |
| [ Generate Guide ]                              |
+------------------------------------------------+
```

### 2.2 Stops List

```text
+------------------------------------------------+
| I-81 Shenandoah - Stops                         |
| 8 stops • 210 km • Wikipedia-cited              |
|                                                |
| 01  "Basalt flows near ..."    0.78 confidence  |
| 02  "Valley and Ridge..."     0.63 confidence  |
| 03  "Karst / caves..."        0.55 confidence  |
| ...                                            |
|                                                |
| [ Export JSON ]  [ Export Markdown ]            |
+------------------------------------------------+
```

### 2.3 Stop Detail

```text
+------------------------------------------------+
| Stop 03 - Karst / caves                         |
|                                                |
| Why this stop                                   |
| - Short grounded explanation...                 |
|                                                |
| What to look for                                |
| - Sinkholes / springs / limestone clues...      |
|                                                |
| Key facts                                       |
| - ...                                           |
|                                                |
| Photo prompts                                   |
| - Wide shot + close-up texture                  |
|                                                |
| Citations                                       |
| - https://en.wikipedia.org/wiki/...             |
+------------------------------------------------+
```

## 3) Camera Flow

### 3.1 Capture

```text
+------------------------------------------------+
| Camera                                          |
|                                                |
|  [ live view ]                                  |
|                                                |
| Tip: take 2 photos                              |
| - Wide context shot                             |
| - Close-up texture shot                         |
|                                                |
| [ Shutter ]   [ Add note ]                      |
+------------------------------------------------+
```

### 3.2 Results (Structured + Narration)

```text
+------------------------------------------------+
| Results                                         |
|                                                |
| Top hypotheses                                  |
| 1) Sandstone (0.62)                             |
| 2) Shale (0.21)                                 |
| 3) Limestone (0.12)                             |
|                                                |
| What I see                                      |
| - Layering (0.74)                               |
| - Grainy texture (0.58)                         |
|                                                |
| Next best photo                                 |
| - Close-up with coin for scale                  |
| - Try shade to reduce glare                     |
|                                                |
| Sources                                         |
| - Wikipedia: Sandstone                          |
|                                                |
| [ Ask follow-up ]   [ Save ]                    |
+------------------------------------------------+
```

## 4) Explore Flow

```text
+------------------------------------------------+
| Explore                                         |
|                                                |
| "You're near <place>. Want to notice:"          |
| - "What landforms shaped this valley?"          |
| - "What trees dominate here?"                   |
| - "Is there karst / caves nearby?"              |
|                                                |
| [ Ask ]                                         |
+------------------------------------------------+
```
