import unittest
import torch
from my_attention_research_project.models.attention.pas_attention import PASAttention

class TestPASAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 10
        self.seq_len_k = 12  # Key length can be different
        self.d_model = 32
        self.top_k_hotspots = 4
        self.num_heads_focused = 4

        self.scanner_config_linear = {
            "type": "PAS_P1_linear",
            "top_k_hotspots": self.top_k_hotspots
        }
        self.focused_attention_config_mha = {
            "num_heads": self.num_heads_focused,
            "dropout_rate": 0.0 # No dropout in tests
        }
        
        self.pas_attention_default = PASAttention(
            d_model=self.d_model,
            scanner_config=self.scanner_config_linear,
            focused_attention_config=self.focused_attention_config_mha,
            dropout_rate=0.0 
        )

        self.query = torch.rand(self.batch_size, self.seq_len_q, self.d_model)
        self.key = torch.rand(self.batch_size, self.seq_len_k, self.d_model)
        self.value = torch.rand(self.batch_size, self.seq_len_k, self.d_model)

    def test_pas_attention_initialization(self):
        self.assertIsNotNone(self.pas_attention_default)
        self.assertTrue(callable(self.pas_attention_default.scanner)) # Check if scanner method is assigned
        self.assertIsNotNone(self.pas_attention_default.focused_attention_mha)
        if self.pas_attention_default.scanner_config.get("type") == "PAS_P1_linear":
            self.assertEqual(self.pas_attention_default.top_k_hotspots, self.top_k_hotspots)

    # 2. Test PAS-P1 Linear Scanner
    def test_linear_scanner_output_shape_and_values(self):
        pas_attention = self.pas_attention_default
        
        # Call scanner directly (it's assigned to self.scanner in __init__)
        # Ensure scanner is actually configured before calling
        if not callable(pas_attention.scanner):
            self.skipTest("Scanner not configured for this PASAttention instance.")

        scanner_output_indices = pas_attention.scanner(self.query, self.key, None)
        
        self.assertEqual(scanner_output_indices.shape, 
                         (self.batch_size, self.seq_len_q, self.top_k_hotspots))
        
        self.assertTrue(torch.all(scanner_output_indices >= 0))
        self.assertTrue(torch.all(scanner_output_indices < self.seq_len_k))

    def test_linear_scanner_padding_masking(self):
        pas_attention = self.pas_attention_default
        if not callable(pas_attention.scanner):
            self.skipTest("Scanner not configured for this PASAttention instance.")

        key_padding_mask = torch.ones(self.batch_size, self.seq_len_k, dtype=torch.bool)
        num_padded_keys = 2
        padded_indices_start = self.seq_len_k - num_padded_keys
        key_padding_mask[:, padded_indices_start:] = False 
        
        hotspot_indices = pas_attention.scanner(self.query, self.key, key_padding_mask)
        
        # Handle case where top_k_hotspots might be 0
        if self.top_k_hotspots == 0:
            self.assertEqual(hotspot_indices.shape[-1], 0)
            return # No indices to check

        for b in range(self.batch_size):
            for q_idx in range(self.seq_len_q):
                for k_idx_pos in range(hotspot_indices.shape[2]): # Iterate up to actual number of hotspots returned
                    selected_key_index = hotspot_indices[b, q_idx, k_idx_pos].item()
                    is_padded = not key_padding_mask[b, selected_key_index].item()
                    self.assertFalse(is_padded, 
                                     f"Hotspot index {selected_key_index} at batch {b}, query_pos {q_idx} "
                                     f"points to a padded key. Padded region starts at {padded_indices_start}.")
    
    # 3. Test Focused Sparse Attention
    def test_focused_attention_output_shape(self):
        pas_attention = self.pas_attention_default
        if not callable(pas_attention.focused_attention):
             self.skipTest("Focused attention not configured for this PASAttention instance.")

        dummy_hotspot_indices = torch.randint(0, self.seq_len_k, 
                                              (self.batch_size, self.seq_len_q, self.top_k_hotspots),
                                              dtype=torch.long)
        if self.top_k_hotspots == 0: # If top_k is 0, dummy_hotspot_indices will have last dim 0
            dummy_hotspot_indices = torch.empty((self.batch_size, self.seq_len_q, 0), dtype=torch.long)


        context = pas_attention.focused_attention(self.query, self.key, self.value, 
                                                  dummy_hotspot_indices, None)
        
        self.assertEqual(context.shape, (self.batch_size, self.seq_len_q, self.d_model))

    def test_focused_attention_masking_gathered_keys(self):
        pas_attention = self.pas_attention_default
        if not callable(pas_attention.focused_attention):
            self.skipTest("Focused attention not configured for this PASAttention instance.")
        if self.top_k_hotspots < 2: # This test assumes at least 2 hotspots can be selected
            self.skipTest("top_k_hotspots is less than 2, skipping this specific masking test.")


        hotspot_indices = torch.zeros(self.batch_size, self.seq_len_q, self.top_k_hotspots, dtype=torch.long)
        
        padded_key_original_index = self.seq_len_k - 1
        hotspot_indices[0, 0, 0] = 0 
        hotspot_indices[0, 0, 1] = padded_key_original_index 
        if self.top_k_hotspots > 1: # ensure other hotspots are valid if they exist
            hotspot_indices[:, :, 2:] = 1 


        original_key_padding_mask = torch.ones(self.batch_size, self.seq_len_k, dtype=torch.bool)
        original_key_padding_mask[:, padded_key_original_index] = False

        distinct_value_marker = torch.ones(self.d_model) * 5.0 
        value_for_test = torch.zeros_like(self.value)
        value_for_test[:, padded_key_original_index, :] = distinct_value_marker
        
        non_padded_key_original_index = 0
        non_padded_value_marker = torch.ones(self.d_model) * 1.0
        value_for_test[:, non_padded_key_original_index, :] = non_padded_value_marker

        context = pas_attention.focused_attention(self.query, self.key, value_for_test, 
                                                  hotspot_indices, original_key_padding_mask)
        
        output_for_q00 = context[0, 0, :]

        similarity_to_padded_value = torch.cosine_similarity(output_for_q00, distinct_value_marker, dim=0)
        
        # If output_for_q00 is all zeros (e.g. if all attended keys ended up being masked, or d_model is 1 and value is 0)
        # then cosine similarity can be nan or 0. We should check this case.
        is_output_zero = torch.allclose(output_for_q00, torch.zeros_like(output_for_q00), atol=1e-6)

        self.assertTrue(is_output_zero or similarity_to_padded_value < 0.5, 
                        f"Output context for query[0,0] seems to reflect contribution from the padded key. "
                        f"Similarity to padded value: {similarity_to_padded_value.item()}. Output sum: {output_for_q00.sum().item()}")

    # 4. Test PASAttention End-to-End
    def test_pas_attention_e2e_run(self):
        pas_attention = self.pas_attention_default
        
        # PASAttention.forward returns: context_vector, hotspot_indices
        context, hotspot_indices_output = pas_attention(self.query, self.key, self.value, None)
        
        self.assertEqual(context.shape, (self.batch_size, self.seq_len_q, self.d_model))
        self.assertIsNotNone(hotspot_indices_output)
        self.assertEqual(hotspot_indices_output.shape, 
                         (self.batch_size, self.seq_len_q, self.top_k_hotspots))

    def test_pas_attention_e2e_with_padding(self):
        pas_attention = self.pas_attention_default
        
        input_padding_mask = torch.ones(self.batch_size, self.seq_len_k, dtype=torch.bool)
        num_padded_keys = 3
        padded_indices_start = self.seq_len_k - num_padded_keys
        input_padding_mask[:, padded_indices_start:] = False
        
        context, hotspot_indices_output = pas_attention(self.query, self.key, self.value, input_padding_mask)
        
        self.assertEqual(context.shape, (self.batch_size, self.seq_len_q, self.d_model))
        self.assertIsNotNone(hotspot_indices_output)
        self.assertEqual(hotspot_indices_output.shape, 
                         (self.batch_size, self.seq_len_q, self.top_k_hotspots))
        
        if self.top_k_hotspots > 0 and hotspot_indices_output.shape[2] > 0 : # if any hotspots were actually returned
            for b in range(self.batch_size):
                for q_idx in range(self.seq_len_q):
                    for k_idx_pos in range(hotspot_indices_output.shape[2]):
                        selected_key_index = hotspot_indices_output[b, q_idx, k_idx_pos].item()
                        is_padded = not input_padding_mask[b, selected_key_index].item()
                        self.assertFalse(is_padded,
                                         f"E2E: Hotspot index {selected_key_index} at batch {b}, query_pos {q_idx} "
                                         f"points to a padded key. Padded region starts at {padded_indices_start}.")

if __name__ == '__main__':
    unittest.main()
