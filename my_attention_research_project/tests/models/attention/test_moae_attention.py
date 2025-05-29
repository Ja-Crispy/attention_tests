import unittest
import torch
import unittest.mock as mock # Added for mocking
from my_attention_research_project.models.attention.moae_attention import MOAEAttention, SoftGateMLP
from my_attention_research_project.models.attention.vanilla_mha import MultiHeadAttention

class TestMOAEAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.batch_size = 2
        self.seq_len = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_expert_configs = [
            {"type": "local_window_mha", "n_heads": 4, "window_size": 5, "dropout_rate": 0.0},
            {"type": "dilated_mha", "n_heads": 2, "dilation_rate": 2, "dropout_rate": 0.0}
        ]
        self.sample_gating_config = {"type": "soft_gate_mlp", "hidden_dim": 64, "dropout_rate": 0.0}

        self.query = torch.rand(self.batch_size, self.seq_len, self.d_model, device=self.device)
        self.key = torch.rand(self.batch_size, self.seq_len, self.d_model, device=self.device)
        self.value = torch.rand(self.batch_size, self.seq_len, self.d_model, device=self.device)
        
        self.moae_dummy_instance = MOAEAttention(
            self.d_model, 
            self.sample_expert_configs, 
            self.sample_gating_config, 
            0.0
        ).to(self.device) # Ensure module is on the correct device
        self.query = self.query.to(self.device)
        self.key = self.key.to(self.device)
        self.value = self.value.to(self.device)


    # 2. Initialization Tests (from previous step)
    def test_initialization_success(self):
        moae_att = MOAEAttention(self.d_model, self.sample_expert_configs, self.sample_gating_config, 0.0)
        self.assertEqual(len(moae_att.experts), len(self.sample_expert_configs))
        self.assertIsInstance(moae_att.experts[0], MultiHeadAttention)
        self.assertEqual(moae_att.experts[0].expert_type, "local_window_mha")
        self.assertEqual(moae_att.experts[0].window_size, 5)
        self.assertIsInstance(moae_att.experts[1], MultiHeadAttention)
        self.assertEqual(moae_att.experts[1].expert_type, "dilated_mha")
        self.assertEqual(moae_att.experts[1].dilation_rate, 2)
        self.assertIsInstance(moae_att.gating_network, SoftGateMLP)

    def test_initialization_invalid_expert_type(self):
        invalid_expert_configs = [{"type": "unknown_expert", "n_heads": 4}]
        with self.assertRaises(ValueError): 
            MOAEAttention(self.d_model, invalid_expert_configs, self.sample_gating_config, 0.0)

    def test_initialization_missing_expert_params(self):
        configs_missing_n_heads = [{"type": "local_window_mha", "window_size": 5}] 
        with self.assertRaises(ValueError):
            MOAEAttention(self.d_model, configs_missing_n_heads, self.sample_gating_config, 0.0)
        configs_missing_window_size = [{"type": "local_window_mha", "n_heads": 4}] 
        with self.assertRaises(ValueError):
            MOAEAttention(self.d_model, configs_missing_window_size, self.sample_gating_config, 0.0)
        configs_missing_dilation_rate = [{"type": "dilated_mha", "n_heads": 2}] 
        with self.assertRaises(ValueError):
            MOAEAttention(self.d_model, configs_missing_dilation_rate, self.sample_gating_config, 0.0)

    def test_initialization_invalid_gating_type(self):
        invalid_gating_config = {"type": "unknown_gating"}
        with self.assertRaises(NotImplementedError): 
            MOAEAttention(self.d_model, self.sample_expert_configs, invalid_gating_config, 0.0)

    def test_initialization_no_experts(self):
        with self.assertRaises(ValueError): 
            MOAEAttention(self.d_model, [], self.sample_gating_config, 0.0)

    # 3. Mask Generation Helper Tests (_generate_local_window_mask) (from previous step)
    def test_local_window_mask_shape_and_type(self):
        mask = self.moae_dummy_instance._generate_local_window_mask(
            self.seq_len, window_size=5, device=self.device, batch_size=self.batch_size)
        self.assertEqual(mask.shape, (self.batch_size, 1, self.seq_len, self.seq_len))
        self.assertEqual(mask.dtype, torch.bool)

    def test_local_window_mask_values(self):
        window_size = 3 
        seq_len_test = 4
        mask = self.moae_dummy_instance._generate_local_window_mask(
            seq_len=seq_len_test, window_size=window_size, device=self.device, batch_size=1)
        self.assertTrue(mask[0, 0, 0, 0]); self.assertTrue(mask[0, 0, 0, 1]); self.assertFalse(mask[0, 0, 0, 2]); self.assertFalse(mask[0, 0, 0, 3])
        self.assertTrue(mask[0, 0, 1, 0]); self.assertTrue(mask[0, 0, 1, 1]); self.assertTrue(mask[0, 0, 1, 2]); self.assertFalse(mask[0, 0, 1, 3])
        self.assertFalse(mask[0, 0, 2, 0]); self.assertTrue(mask[0, 0, 2, 1]); self.assertTrue(mask[0, 0, 2, 2]); self.assertTrue(mask[0, 0, 2, 3])
        self.assertFalse(mask[0, 0, 3, 0]); self.assertFalse(mask[0, 0, 3, 1]); self.assertTrue(mask[0, 0, 3, 2]); self.assertTrue(mask[0, 0, 3, 3])

    def test_local_window_mask_with_padding(self):
        seq_len_test = 4; window_size = 3
        input_pad = torch.tensor([[True, True, False, False]], device=self.device, dtype=torch.bool) 
        mask = self.moae_dummy_instance._generate_local_window_mask(
            seq_len_test, window_size, self.device, 1, input_pad)
        self.assertFalse(torch.any(mask[0, 0, :, 2])); self.assertFalse(torch.any(mask[0, 0, :, 3]))
        self.assertTrue(mask[0,0,0,0]); self.assertTrue(mask[0,0,0,1]); self.assertFalse(mask[0,0,0,2])
        self.assertTrue(mask[0,0,1,0]); self.assertTrue(mask[0,0,1,1]); self.assertFalse(mask[0,0,1,2])

    # 4. Mask Generation Helper Tests (_generate_dilated_mask) (from previous step)
    def test_dilated_mask_shape_and_type(self):
        mask = self.moae_dummy_instance._generate_dilated_mask(
            self.seq_len, dilation_rate=2, device=self.device, batch_size=self.batch_size)
        self.assertEqual(mask.shape, (self.batch_size, 1, self.seq_len, self.seq_len))
        self.assertEqual(mask.dtype, torch.bool)

    def test_dilated_mask_values(self):
        dilation_rate = 2; seq_len_test = 5 
        mask = self.moae_dummy_instance._generate_dilated_mask(seq_len_test, dilation_rate, self.device, 1)
        expected_key_pattern = torch.tensor([True, False, True, False, True], device=self.device)
        for i in range(seq_len_test): self.assertTrue(torch.equal(mask[0, 0, i, :], expected_key_pattern))

    def test_dilated_mask_with_padding(self):
        seq_len_test = 5; dilation_rate = 2
        input_pad = torch.tensor([[True, True, False, True, False]], device=self.device, dtype=torch.bool) 
        mask = self.moae_dummy_instance._generate_dilated_mask(seq_len_test, dilation_rate, self.device, 1, input_pad)
        expected_key_pattern_after_padding = torch.tensor([True, False, False, False, False], device=self.device)
        for i in range(seq_len_test): self.assertTrue(torch.equal(mask[0, 0, i, :], expected_key_pattern_after_padding))

    # 5. Gating Mechanism (SoftGateMLP) Tests
    def test_soft_gate_mlp_output(self):
        num_experts_test = len(self.sample_expert_configs)
        gate_mlp = SoftGateMLP(embed_dim=self.d_model, num_experts=num_experts_test, hidden_dim=64).to(self.device)
        gate_weights = gate_mlp(self.query)
        
        self.assertEqual(gate_weights.shape, (self.batch_size, self.seq_len, num_experts_test))
        self.assertTrue(torch.allclose(gate_weights.sum(dim=-1), torch.ones_like(gate_weights.sum(dim=-1))))
        self.assertTrue(torch.all(gate_weights >= 0) and torch.all(gate_weights <= 1))

    # 6. Individual Expert Forward Pass (within MOAEAttention context)
    def test_expert_local_window_forward(self):
        config = [{"type": "local_window_mha", "n_heads": 2, "window_size": 3, "dropout_rate": 0.0}]
        gating_cfg = {"type": "soft_gate_mlp", "hidden_dim": 32, "dropout_rate": 0.0}
        moae_att = MOAEAttention(self.d_model, config, gating_cfg, 0.0).to(self.device)
        
        expert = moae_att.experts[0]
        mask = moae_att._generate_local_window_mask(
            self.seq_len, expert.window_size, self.device, self.batch_size)
        output, _ = expert(self.query, self.key, self.value, mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_expert_dilated_mha_forward(self):
        config = [{"type": "dilated_mha", "n_heads": 2, "dilation_rate": 2, "dropout_rate": 0.0}]
        gating_cfg = {"type": "soft_gate_mlp", "hidden_dim": 32, "dropout_rate": 0.0}
        moae_att = MOAEAttention(self.d_model, config, gating_cfg, 0.0).to(self.device)

        expert = moae_att.experts[0]
        mask = moae_att._generate_dilated_mask(
            self.seq_len, expert.dilation_rate, self.device, self.batch_size)
        output, _ = expert(self.query, self.key, self.value, mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    # 7. MOAEAttention Full Forward Pass Tests
    def test_moae_forward_pass_shape(self):
        moae_att = self.moae_dummy_instance # Uses self.sample_expert_configs
        context = moae_att(self.query, self.key, self.value, None)
        self.assertEqual(context.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_moae_forward_pass_with_padding(self):
        moae_att = self.moae_dummy_instance
        input_pad = torch.ones(self.batch_size, self.seq_len, device=self.device, dtype=torch.bool)
        input_pad[:, -self.seq_len//2:] = False # Pad last half of keys/values
        
        context = moae_att(self.query, self.key, self.value, input_padding_mask=input_pad)
        self.assertEqual(context.shape, (self.batch_size, self.seq_len, self.d_model))
        # Further checks for padding influence are complex without known weights/outputs

    def test_moae_forward_combination_logic(self):
        single_expert_configs = [self.sample_expert_configs[0].copy()] # Local window expert
        # Ensure gating config is compatible with 1 expert for SoftGateMLP
        single_expert_gating_cfg = {"type": "soft_gate_mlp", "hidden_dim": 32, "dropout_rate": 0.0}
        
        moae_att = MOAEAttention(self.d_model, single_expert_configs, single_expert_gating_cfg, 0.0).to(self.device)

        # Mock gating_network to return all weight to the single expert
        # Softmax on [anything] for a single expert will result in [1.0]
        moae_att.gating_network = mock.Mock(return_value=torch.ones(self.batch_size, self.seq_len, 1, device=self.device))
        
        expert_module = moae_att.experts[0]
        the_mask_for_expert_0 = moae_att._generate_local_window_mask(
            self.seq_len, expert_module.window_size, self.device, self.batch_size, None)
        
        expert_direct_output, _ = expert_module(self.query, self.key, self.value, mask=the_mask_for_expert_0)
        moae_output = moae_att(self.query, self.key, self.value, None)
        
        self.assertTrue(torch.allclose(moae_output, expert_direct_output, atol=1e-6))

    # 8. Configuration Flexibility & Edge Case Tests
    def test_moae_seq_len_1(self):
        query_s1 = torch.randn(self.batch_size, 1, self.d_model, device=self.device)
        key_s1 = torch.randn(self.batch_size, 1, self.d_model, device=self.device)
        value_s1 = torch.randn(self.batch_size, 1, self.d_model, device=self.device)
        
        moae_att = self.moae_dummy_instance # Uses standard sample configs
        context = moae_att(query_s1, key_s1, value_s1, None)
        self.assertEqual(context.shape, (self.batch_size, 1, self.d_model))

    def test_moae_different_num_experts(self):
        expert_configs_3 = self.sample_expert_configs + [{"type": "local_window_mha", "n_heads": 2, "window_size": 7, "dropout_rate": 0.0}]
        # Gating config needs to be aware of num_experts, but SoftGateMLP in MOAEAttention init calculates it
        moae_att = MOAEAttention(self.d_model, expert_configs_3, self.sample_gating_config, 0.0).to(self.device)
        
        self.assertEqual(len(moae_att.experts), 3)
        context = moae_att(self.query, self.key, self.value, None)
        self.assertEqual(context.shape, (self.batch_size, self.seq_len, self.d_model))

if __name__ == '__main__':
    unittest.main()
