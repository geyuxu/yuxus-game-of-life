# Digital Primordial Soup

A GPU-accelerated artificial life simulation where neural network-driven organisms compete for survival through predation, reproduction, and evolution.

## Features

- **Neuroevolution**: Each organism has its own neural network that controls behavior. Weights are inherited and mutated during reproduction
- **Reinforcement Learning**: Neural networks are fine-tuned within each generation based on survival rewards
- **Dynamic Speciation**: When one species dominates (>75%), it automatically splits into a new species to maintain ecosystem diversity
- **GPU Acceleration**: All computations run on CUDA/MPS/CPU via PyTorch for real-time simulation
- **Network Persistence**: Best-performing neural networks are saved and loaded across sessions

## Demo

![Simulation Screenshot](docs/demo.png)

The visualization shows:
- **Top chart**: Global population dynamics over all generations
- **Bottom left**: Real-time grid view of organisms (colors = species, brightness = energy)
- **Bottom right**: Recent species population trends

## Installation

```bash
# Clone the repository
git clone https://github.com/geyuxu/digital-primordial-soup.git
cd digital-primordial-soup

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run simulation (default 20x20 grid)
python main.py

# Run with larger grid
python main.py --grid 30

# Validation mode: test trained network vs random networks
python main.py --validate
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--grid`, `-g` | Grid size (default: 20) |
| `--validate`, `-v` | Validation mode: S0 uses trained weights, S1/S2 use random |

## How It Works

### Species & Predation
- All species start with identical parameters
- Differentiation emerges purely from neural network weights
- Every species can prey on every other species (50% success rate)
- Successful hunts grant 120% of prey's energy

### Neural Network
- **Input (20 neurons)**: Neighbor energy levels, same/different species counts, own energy
- **Hidden (8 neurons)**: Fully connected with tanh activation
- **Output (7 actions)**: Stay, Move (4 directions), Eat, Reproduce

### Evolution Mechanisms
1. **Inheritance**: Offspring inherit parent's neural network with small mutations
2. **RL Fine-tuning**: Successful actions (hunting, escaping, reproducing) reinforce neural network weights
3. **Speciation**: Dominant species split to prevent competitive exclusion

### Fitness & Persistence
- Fitness = lifetime + reproduction_count × 10
- Best network saved every 50 generations to `best_brain.pt`
- 20% of initial cells inherit saved weights on startup

## Configuration

Key parameters in `main.py`:

```python
# Simulation
GRID_SIZE = 20
INITIAL_NUM_SPECIES = 3
MAX_SPECIES = 10

# Evolution
MUTATION_RATE = 0.1
DOMINANCE_THRESHOLD = 0.75
SPLIT_MUTATION_RATE = 0.3

# Reinforcement Learning
RL_LEARNING_RATE = 0.01
REWARD_EAT_PREY = 2.0
REWARD_SURVIVE_ATTACK = 1.0
REWARD_REPRODUCE = 1.5
```

## Project Structure

```
digital-primordial-soup/
├── main.py           # Main simulation code
├── best_brain.pt     # Saved neural network weights
├── requirements.txt  # Python dependencies
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA/MPS support recommended)
- NumPy
- Matplotlib

## License

MIT License

---

# 数字原始汤

一个 GPU 加速的人工生命模拟器，神经网络驱动的生物通过捕食、繁殖和进化竞争生存。

## 特性

- **神经进化**：每个生物都有自己的神经网络控制行为，权重在繁殖时遗传和变异
- **强化学习**：神经网络根据生存奖励在每一代内进行微调
- **动态物种形成**：当某物种占比超过 75% 时自动分裂，维持生态系统多样性
- **GPU 加速**：所有计算通过 PyTorch 在 CUDA/MPS/CPU 上运行
- **网络持久化**：最优神经网络跨会话保存和加载

## 运行方式

```bash
# 默认模式
python main.py

# 大网格
python main.py --grid 30

# 验证模式：测试训练网络 vs 随机网络
python main.py --validate
```

## 工作原理

### 物种与捕食
- 所有物种参数完全相同
- 差异仅来自神经网络权重
- 任意物种可捕食其他物种（50% 成功率）

### 进化机制
1. **遗传**：后代继承父代神经网络 + 小变异
2. **强化学习**：成功行为（捕猎、逃脱、繁殖）强化神经网络
3. **物种分裂**：优势物种自动分裂防止竞争排斥

### 适应度与持久化
- 适应度 = 存活代数 + 繁殖次数 × 10
- 每 50 代保存最优网络到 `best_brain.pt`
- 启动时 20% 的细胞继承保存的权重
