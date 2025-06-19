# Encoding Analysis Report

Dataset: `data/dataset.csv`  
Total samples: 607225  
Analysis date: 2025-06-16 20:28:19  

## Summary

| Encoding Type | Min Length | Max Length | Mean Length | Length Consistency |
|---------------|------------|------------|-------------|-------------------|
| 1-mer | 4 | 4 | 4.0 | 100.0% (4) |

## 1-mer Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=4, Max=4, Mean=4.0
- **Length consistency**: 100.0% have length 4
- **Value range**: [9.0, 178.0]

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 4 | 607225 | 100.00% |

### Example Encoding

```
[ 88. 135.  51.  39.]
```

| 2-mer | 16 | 16 | 16.0 | 100.0% (16) |

## 2-mer Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=16, Max=16, Mean=16.0
- **Length consistency**: 100.0% have length 16
- **Value range**: [0.0, 103.0]

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 16 | 607225 | 100.00% |

### Example Encoding

```
[18. 40. 14. 16. 37. 63. 22. 12. 19. 23.  6.  3. 13.  9.  9.  8.]
```

| 3-mer | 64 | 64 | 64.0 | 100.0% (64) |

## 3-mer Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=64, Max=64, Mean=64.0
- **Length consistency**: 100.0% have length 64
- **Value range**: [0.0, 67.0]

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 64 | 607225 | 100.00% |

### Example Encoding

```
[ 2. 11.  4.  1.  7. 20. 12.  1.  8.  3.  2.  1.  3.  3.  5.  5.  7. 16.
  8.  6.]...
```

| 4-mer | 256 | 256 | 256.0 | 100.0% (256) |

## 4-mer Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=256, Max=256, Mean=256.0
- **Length consistency**: 100.0% have length 256
- **Value range**: [0.0, 46.0]

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 256 | 607225 | 100.00% |

### Example Encoding

```
[0. 2. 0. 0. 4. 4. 3. 0. 4. 0. 0. 0. 0. 1. 0. 0. 1. 3. 2. 1.]...
```

| 5-mer | 1024 | 1024 | 1024.0 | 100.0% (1024) |

## 5-mer Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=1024, Max=1024, Mean=1024.0
- **Length consistency**: 100.0% have length 1024
- **Value range**: [0.0, 31.0]

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 1024 | 607225 | 100.00% |

### Example Encoding

```
[0. 0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 2. 1. 0.]...
```

| 1-bit | 320 | 320 | 320.0 | 100.0% (320) |

## 1-bit Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=320, Max=320, Mean=320.0
- **Length consistency**: 100.0% have length 320
- **All values are binary (0 or 1)**: False

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 320 | 607225 | 100.00% |

### Example Encoding

```
[1. 2. 3. 3. 3. 2. 1. 3. 2. 3. 1. 1. 3. 1. 3. 3. 4. 2. 3. 2.]...
```

| 2-bit | 640 | 640 | 640.0 | 100.0% (640) |

## 2-bit Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=640, Max=640, Mean=640.0
- **Length consistency**: 100.0% have length 640
- **All values are binary (0 or 1)**: True

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 640 | 607225 | 100.00% |

### Example Encoding

```
[0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0.]...
```

| 3-bit | 960 | 960 | 960.0 | 100.0% (960) |

## 3-bit Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=960, Max=960, Mean=960.0
- **Length consistency**: 100.0% have length 960
- **All values are binary (0 or 1)**: True

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 960 | 607225 | 100.00% |

### Example Encoding

```
[1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 1. 0. 1. 0.]...
```

| 4-bit | 1280 | 1280 | 1280.0 | 100.0% (1280) |

## 4-bit Encoding

- **Samples processed**: 607225
- **Encoding length**: Min=1280, Max=1280, Mean=1280.0
- **Length consistency**: 100.0% have length 1280
- **All values are binary (0 or 1)**: True

### Length Distribution

| Length | Count | Percentage |
|--------|-------|------------|
| 1280 | 607225 | 100.00% |

### Example Encoding

```
[1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0.]...
```

| 4-row matrix | 299 | 320 | 311.6 | 73.5% (313) |

## 4-row Matrix Encoding

- **Samples processed**: 607225
- **Consistent row lengths**: 607225 out of 607225 (100.0%)
- **Matrix width**: Min=299, Max=320, Mean=311.6
- **Width consistency**: 73.5% have width 313
- **All values are binary (0 or 1)**: True

### Width Distribution

| Width | Count | Percentage |
|-------|-------|------------|
| 313 | 446486 | 73.53% |
| 310 | 48076 | 7.92% |
| 303 | 20973 | 3.45% |
| 304 | 20828 | 3.43% |
| 312 | 17215 | 2.84% |
| 311 | 8449 | 1.39% |
| 307 | 8055 | 1.33% |
| 309 | 6586 | 1.08% |
| 306 | 5293 | 0.87% |
| 301 | 4848 | 0.80% |

### Example Matrix

First 10 positions of first sample:

```
Row 1: [1. 0. 0. 0. 0. 0. 1. 0. 0. 0.]...
Row 2: [0. 1. 0. 0. 0. 1. 0. 0. 1. 0.]...
Row 3: [0. 0. 1. 1. 1. 0. 0. 1. 0. 1.]...
Row 4: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]...
```

