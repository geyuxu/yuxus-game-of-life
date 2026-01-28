# Changelog

All notable changes to Neuroevolution Arena will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- Reverted genome distance calculation from cosine similarity back to Euclidean distance
  - Cosine similarity with 0.8 threshold was too strict, causing extinction
  - Euclidean distance with 0.5 threshold allows proper mating and reproduction

### Changed
- Increased `ELITE_RATIO` from 0.01 to 0.15 for better trained cell expansion

## [2.2.1] - 2025-01-28

### Fixed
- **CRITICAL**: Remove O(N²) diversity calculation from `_update_best_network()` that caused severe lag
- **PERFORMANCE**: Reduce history stats update frequency (every 10 generations instead of every step)
- **UI**: Optimize stats panel layout to prevent text overlap at high population counts
- **VISUAL**: Fix color jumping by reducing color update interval from 200 to 10 generations

## [2.2.0] - 2025-01-27

### Added
- Complete checkpoint save/load system (Ctrl+S / Ctrl+L)
- Real-time parameter adjustment UI (keys 1-4 + arrow keys)
- Genome heatmap analysis visualization (press H)
- A/B testing framework for parameter comparison
- Lineage tracking: Gen0 (trained) vs descendants vs random
- Experience replay buffer for stable RL learning

### Changed
- Fitness function now multi-objective: lifetime + reproduction + diversity + energy
- Chemical signaling system with 4 chemical types
- Genome-based species identification (12-dimensional genome fingerprint)

### Performance
- GPU-accelerated simulation with CUDA tensors
- Pre-allocated buffers to reduce memory allocation
- Optimized neural network batch forward pass

## [2.1.0] - 2025-01-26

### Added
- Pygame-based high-performance renderer (60+ FPS)
- Camera zoom and pan controls
- Chemical field overlay visualization
- Grid overlay toggle

### Changed
- Migrated from Matplotlib to Pygame for real-time rendering
- Improved cell rendering with anti-aliasing

## [2.0.0] - 2025-01-25

### Added
- Neural network controlled organisms
- Policy gradient reinforcement learning
- Genome-based mating system
- Species clustering and visualization

### Changed
- Complete rewrite from cellular automata to neuroevolution system
- Each cell has individual neural network weights

## [1.0.0] - 2025-01-20

### Added
- Initial release
- Basic Game of Life simulation
- Simple species system
- Energy and reproduction mechanics
