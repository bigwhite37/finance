#!/usr/bin/env python3
"""
Transformer配置生成器

根据不同场景生成推荐的Transformer配置
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.models.transformer import TransformerConfig
from rl_trading_system.models.sac_agent import SACConfig


def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """获取预设配置"""
    return {
        "lightweight": {
            "name": "轻量配置",
            "description": "适用于资源受限环境，快速训练和推理",
            "transformer": {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 3,
                "d_ff": 512,
                "dropout": 0.1,
                "max_seq_len": 252,
                "n_features": 37,
                "activation": "gelu"
            },
            "sac": {
                "state_dim": 128,
                "action_dim": 3,
                "hidden_dim": 256,
                "use_transformer": True
            }
        },
        "standard": {
            "name": "标准配置",
            "description": "平衡性能与资源消耗，推荐用于大部分场景",
            "transformer": {
                "d_model": 256,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 1024,
                "dropout": 0.1,
                "max_seq_len": 252,
                "n_features": 37,
                "activation": "gelu"
            },
            "sac": {
                "state_dim": 256,
                "action_dim": 3,
                "hidden_dim": 512,
                "use_transformer": True
            }
        },
        "high_performance": {
            "name": "高性能配置",
            "description": "适用于GPU充足环境，追求最佳性能",
            "transformer": {
                "d_model": 512,
                "n_heads": 16,
                "n_layers": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "max_seq_len": 252,
                "n_features": 37,
                "activation": "gelu"
            },
            "sac": {
                "state_dim": 512,
                "action_dim": 3,
                "hidden_dim": 1024,
                "use_transformer": True
            }
        },
        "debug": {
            "name": "调试配置",
            "description": "极小配置，用于快速调试和测试",
            "transformer": {
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "d_ff": 256,
                "dropout": 0.1,
                "max_seq_len": 60,
                "n_features": 37,
                "activation": "gelu"
            },
            "sac": {
                "state_dim": 64,
                "action_dim": 3,
                "hidden_dim": 128,
                "use_transformer": True
            }
        }
    }


def validate_config(transformer_config: Dict, sac_config: Dict) -> bool:
    """验证配置的合理性"""
    try:
        # 基本验证
        d_model = transformer_config["d_model"]
        n_heads = transformer_config["n_heads"]
        
        assert d_model > 0, "d_model必须大于0"
        assert n_heads > 0, "n_heads必须大于0"
        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被n_heads ({n_heads}) 整除"
        
        # 与SAC配置的一致性
        assert sac_config["state_dim"] == d_model, f"SAC state_dim ({sac_config['state_dim']}) 必须与Transformer d_model ({d_model}) 一致"
        
        # 特征维度检查
        assert transformer_config["n_features"] == 37, "当前系统要求n_features=37"
        
        # 合理性检查
        assert transformer_config["dropout"] <= 0.5, "dropout不应超过0.5"
        assert transformer_config["d_ff"] >= d_model, "d_ff应该至少等于d_model"
        
        return True
    except AssertionError as e:
        print(f"❌ 配置验证失败: {e}")
        return False


def estimate_model_complexity(config: Dict[str, Any]) -> Dict[str, Any]:
    """估算模型复杂度"""
    transformer = config["transformer"]
    
    d_model = transformer["d_model"]
    n_layers = transformer["n_layers"]
    n_heads = transformer["n_heads"]
    d_ff = transformer["d_ff"]
    n_features = transformer["n_features"]
    max_seq_len = transformer["max_seq_len"]
    
    # 估算参数量
    # 输入投影层
    input_proj_params = n_features * d_model
    
    # Transformer层参数
    # 每层包含：多头注意力 + 前馈网络 + 层归一化
    per_layer_params = (
        4 * d_model * d_model +  # Q, K, V, O 投影
        2 * d_model * d_ff +     # 前馈网络
        4 * d_model              # 层归一化参数
    )
    transformer_params = n_layers * per_layer_params
    
    # 输出投影层
    output_proj_params = d_model * d_model
    
    total_params = input_proj_params + transformer_params + output_proj_params
    
    # 估算内存使用 (假设float32)
    memory_mb = total_params * 4 / (1024 * 1024)
    
    # 估算训练时内存 (包含梯度和优化器状态)
    training_memory_mb = memory_mb * 4
    
    # 复杂度等级
    if total_params < 1e6:
        complexity = "低"
    elif total_params < 5e6:
        complexity = "中"
    else:
        complexity = "高"
    
    return {
        "total_parameters": total_params,
        "model_size_mb": memory_mb,
        "training_memory_mb": training_memory_mb,
        "complexity": complexity,
        "estimated_training_time": f"{'快' if total_params < 1e6 else '中' if total_params < 5e6 else '慢'}",
        "inference_speed": f"{'很快' if total_params < 1e6 else '快' if total_params < 5e6 else '中等'}"
    }


def generate_python_code(config: Dict[str, Any]) -> str:
    """生成Python代码"""
    transformer = config["transformer"]
    sac = config["sac"]
    
    code = f'''import torch
from rl_trading_system.models.transformer import TransformerConfig
from rl_trading_system.models.sac_agent import SACConfig, SACAgent

# Transformer配置
transformer_config = TransformerConfig(
    d_model={transformer["d_model"]},
    n_heads={transformer["n_heads"]},
    n_layers={transformer["n_layers"]},
    d_ff={transformer["d_ff"]},
    dropout={transformer["dropout"]},
    max_seq_len={transformer["max_seq_len"]},
    n_features={transformer["n_features"]},
    activation="{transformer["activation"]}"
)

# SAC配置
sac_config = SACConfig(
    state_dim={sac["state_dim"]},
    action_dim={sac["action_dim"]},
    hidden_dim={sac["hidden_dim"]},
    use_transformer={str(sac["use_transformer"]).title()},
    transformer_config=transformer_config,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 创建智能体
agent = SACAgent(sac_config)

print(f"✅ 创建了{{config['name']}}智能体")
print(f"   Transformer参数: {{transformer_config.d_model}}维度, {{transformer_config.n_heads}}头, {{transformer_config.n_layers}}层")
print(f"   设备: {{sac_config.device}}")
'''
    return code


def save_config(config: Dict[str, Any], output_path: Path, format: str):
    """保存配置文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif format == "yaml":
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif format == "python":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generate_python_code(config))


def main():
    parser = argparse.ArgumentParser(description="Transformer配置生成器")
    parser.add_argument("--preset", type=str, 
                       choices=["lightweight", "standard", "high_performance", "debug"],
                       default="standard",
                       help="预设配置类型")
    parser.add_argument("--output", type=str, default="config/generated_transformer_config",
                       help="输出文件路径（不含扩展名）")
    parser.add_argument("--format", type=str, choices=["json", "yaml", "python"], 
                       default="yaml", help="输出格式")
    parser.add_argument("--list-presets", action="store_true",
                       help="列出所有预设配置")
    parser.add_argument("--validate", action="store_true",
                       help="验证配置合理性")
    parser.add_argument("--analyze", action="store_true",
                       help="分析模型复杂度")
    
    # 自定义参数
    parser.add_argument("--d-model", type=int, help="自定义模型维度")
    parser.add_argument("--n-heads", type=int, help="自定义注意力头数")
    parser.add_argument("--n-layers", type=int, help="自定义层数")
    parser.add_argument("--d-ff", type=int, help="自定义前馈网络维度")
    
    args = parser.parse_args()
    
    presets = get_preset_configs()
    
    if args.list_presets:
        print("📋 可用的预设配置：\n")
        for key, preset in presets.items():
            print(f"🔹 {key}: {preset['name']}")
            print(f"   描述: {preset['description']}")
            transformer = preset["transformer"]
            print(f"   参数: d_model={transformer['d_model']}, n_heads={transformer['n_heads']}, n_layers={transformer['n_layers']}")
            print()
        return
    
    # 获取基础配置
    if args.preset in presets:
        config = presets[args.preset].copy()
    else:
        print(f"❌ 未知的预设配置: {args.preset}")
        print("可用配置:", list(presets.keys()))
        return
    
    # 应用自定义参数
    if args.d_model:
        config["transformer"]["d_model"] = args.d_model
        config["sac"]["state_dim"] = args.d_model  # 保持一致
    if args.n_heads:
        config["transformer"]["n_heads"] = args.n_heads
    if args.n_layers:
        config["transformer"]["n_layers"] = args.n_layers
    if args.d_ff:
        config["transformer"]["d_ff"] = args.d_ff
    
    print(f"🔧 生成配置: {config['name']}")
    print(f"📝 描述: {config['description']}")
    print()
    
    # 验证配置
    if args.validate or args.analyze:
        is_valid = validate_config(config["transformer"], config["sac"])
        if not is_valid:
            return
        print("✅ 配置验证通过")
        print()
    
    # 分析复杂度
    if args.analyze:
        complexity = estimate_model_complexity(config)
        print("📊 模型复杂度分析:")
        print(f"   参数量: {complexity['total_parameters']:,}")
        print(f"   模型大小: {complexity['model_size_mb']:.1f} MB")
        print(f"   训练内存: {complexity['training_memory_mb']:.1f} MB")
        print(f"   复杂度: {complexity['complexity']}")
        print(f"   训练速度: {complexity['estimated_training_time']}")
        print(f"   推理速度: {complexity['inference_speed']}")
        print()
    
    # 保存配置
    output_path = Path(args.output).with_suffix(f".{args.format}")
    save_config(config, output_path, args.format)
    
    print(f"💾 配置已保存到: {output_path}")
    
    # 显示使用方法
    if args.format == "python":
        print("\n📖 使用方法:")
        print(f"   python -c \"exec(open('{output_path}').read())\"")
    elif args.format == "yaml":
        print("\n📖 使用方法:")
        print(f"   在代码中: config = yaml.load(open('{output_path}'))")
    elif args.format == "json":
        print("\n📖 使用方法:")
        print(f"   在代码中: config = json.load(open('{output_path}'))")


if __name__ == "__main__":
    main()