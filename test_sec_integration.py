#!/usr/bin/env python3
"""Test SEC curriculum integration with VERL."""

import sys
import os
sys.path.append('/Users/arc-aman/Documents/GitHub/verl')

def test_sec_curriculum():
    """Test SEC curriculum class."""
    from verl.trainer.ppo.core_algos import SECCurriculum
    
    # Initialize curriculum
    curriculum = SECCurriculum(alpha=0.3, tau=0.5)
    print("✓ SEC curriculum initialized")
    
    # Test arm selection
    arm = curriculum.select_arm()
    print(f"✓ Selected arm: {arm}")
    
    # Test Q-value update
    curriculum.update_q_values(arm, 0.75)
    print(f"✓ Updated Q-values: max={curriculum.q_values.max():.3f}")
    
    # Test metrics
    metrics = curriculum.get_metrics()
    print(f"✓ Metrics: {metrics}")
    
    return True

def test_config_loading():
    """Test SEC curriculum config loading."""
    import yaml
    
    with open('sec_curriculum_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert config['data']['use_sec_curriculum'] == True
    assert config['data']['sec_alpha'] == 0.3
    assert config['data']['train_batch_size'] == 256
    assert config['algorithm']['adv_estimator'] == 'grpo'
    
    print("✓ SEC curriculum config validated")
    return True

def test_sampler_creation():
    """Test SEC sampler creation logic."""
    # Mock data config
    class MockDataConfig:
        def __init__(self):
            self.use_sec_curriculum = True
            self.sec_alpha = 0.3
            self.sec_tau = 0.5
        
        def get(self, key, default):
            return getattr(self, key.replace('sec_', ''), default)
    
    # Mock dataset
    class MockDataset:
        def __len__(self):
            return 1000
    
    from verl.trainer.main_ppo import create_rl_sampler, SECCurriculumSampler
    
    config = MockDataConfig()
    dataset = MockDataset()
    
    sampler = create_rl_sampler(config, dataset)
    
    assert isinstance(sampler, SECCurriculumSampler)
    assert hasattr(dataset, 'sec_curriculum')
    
    print("✓ SEC curriculum sampler created successfully")
    return True

def main():
    """Run all tests."""
    print("Testing SEC curriculum integration with VERL...")
    print("=" * 50)
    
    try:
        test_sec_curriculum()
        test_config_loading()
        test_sampler_creation()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED - SEC curriculum integration successful!")
        print("\nTo run training with SEC curriculum:")
        print("cd /Users/arc-aman/Documents/GitHub/verl")
        print("python -m verl.trainer.main_ppo --config-path=. --config-name=sec_curriculum_config")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)