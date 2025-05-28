import torch
import pytest
from my_attention_research_project.models.attention.ahc_attention import AHCAttention

class TestAHCAttentionMasking:
    def _create_ahc_attention(self, d_model=4, n_heads=2, chunk_size=2, 
                              summarization_type="S1_pool", pool_type="mean", 
                              global_attention_type="G1_full_mha", combination_type="C1_broadcast_add"):
        summarization_config = {"type": summarization_type}
        if summarization_type in ["S1_pool", "S2_linear_on_pool"]:
            if pool_type is None: # Ensure pool_type is provided if summarization needs it
                 raise ValueError(f"pool_type must be specified for summarization type {summarization_type}")
            summarization_config["pool_type"] = pool_type
        
        global_attention_config = {"type": global_attention_type}
        if global_attention_type == "G1_full_mha": # num_heads is expected inside the config for this type
            global_attention_config["num_heads"] = n_heads
        # If global_attention_type is None, no other params are needed in its config.

        combination_config = {"type": combination_type}
        
        return AHCAttention(
            d_model=d_model,
            n_heads=n_heads, # n_heads for local_mha
            chunk_size=chunk_size,
            summarization_method_config=summarization_config,
            global_attention_method_config=global_attention_config,
            combination_method_config=combination_config,
            dropout_rate=0.0
        )

    # Test Case 1: No Mask
    def test_no_mask(self):
        batch_size, seq_len, d_model = 1, 4, 4
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=2)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, _ = ahc_attention.forward(query, key, value, input_padding_mask=None)
        assert output.shape == (batch_size, seq_len, d_model)

    # Test Case 2: Full Padding Mask
    def test_full_padding_mask(self):
        batch_size, seq_len, d_model = 1, 4, 4
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=2)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        input_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool) 

        output, _ = ahc_attention.forward(query, key, value, input_padding_mask=input_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)

    # Test Case 3: Partial Mask - Within a Chunk
    def test_partial_mask_within_chunk(self):
        batch_size, seq_len, d_model = 1, 4, 4
        chunk_size = 2
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=chunk_size)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        input_padding_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)
        
        output, _ = ahc_attention.forward(query, key, value, input_padding_mask=input_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)

    # Test Case 4: Partial Mask - Across Chunk Boundaries
    def test_partial_mask_across_chunks(self):
        batch_size, seq_len, d_model = 1, 4, 4
        chunk_size = 2
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=chunk_size)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        input_padding_mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)
        
        output, _ = ahc_attention.forward(query, key, value, input_padding_mask=input_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)

    # Test Case 5: Mask Affecting Summary Tokens
    def test_mask_affecting_summary(self):
        batch_size, seq_len, d_model = 1, 4, 4
        chunk_size = 2
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=chunk_size)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        input_padding_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
        
        output, _ = ahc_attention.forward(query, key, value, input_padding_mask=input_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)

class TestAHCAttentionStaticHelpers(TestAHCAttentionMasking): # Inherit helper
    @pytest.mark.parametrize("mask_input, target_seq_len, device_str, expected_shape, expect_error", [
        (None, 4, "cpu", None, None), # None mask
        (torch.ones(1, 4, dtype=torch.bool), 4, "cpu", (1, 1, 1, 4), None), # 2D bool
        (torch.ones(1, 4, dtype=torch.float), 4, "cpu", (1, 1, 1, 4), None), # 2D float
        (torch.ones(1, 1, 1, 4, dtype=torch.float), 4, "cpu", (1, 1, 1, 4), None), # 4D float (already processed)
        (torch.ones(1, 3, dtype=torch.bool), 4, "cpu", None, ValueError), # Incorrect 2D shape
        (torch.ones(1, 1, 2, 4, dtype=torch.float), 4, "cpu", None, ValueError), # Incorrect 4D shape (Q!=1)
        (torch.ones(1, 1, 1, 3, dtype=torch.float), 4, "cpu", None, ValueError), # Incorrect 4D shape (S!=target_S)
        (torch.ones(1, 4, 4, dtype=torch.float), 4, "cpu", None, ValueError), # Unsupported 3D
        (torch.ones(1, dtype=torch.float), 4, "cpu", None, ValueError), # Unsupported 1D
    ])
    def test_process_input_mask(self, mask_input, target_seq_len, device_str, expected_shape, expect_error):
        device = torch.device(device_str)
        original_mask_input_for_value_check = None
        if mask_input is not None:
            mask_input_on_device = mask_input.clone().to(device)
            if mask_input.dtype == torch.bool: # Keep original boolean for value check
                 original_mask_input_for_value_check = mask_input.clone()
        else:
            mask_input_on_device = None

        if expect_error:
            with pytest.raises(expect_error):
                AHCAttention._process_input_mask(mask_input_on_device, target_seq_len, device)
        else:
            processed_mask = AHCAttention._process_input_mask(mask_input_on_device, target_seq_len, device)
            if expected_shape is None:
                assert processed_mask is None
            else:
                assert processed_mask.shape == expected_shape
                assert processed_mask.dtype == torch.float
                assert processed_mask.device == device
                if original_mask_input_for_value_check is not None : # Check content for boolean inputs
                    expected_float_content = original_mask_input_for_value_check.float()
                    if original_mask_input_for_value_check.ndim == 2:
                        expected_float_content = expected_float_content.unsqueeze(1).unsqueeze(2)
                    assert torch.equal(processed_mask, expected_float_content.to(device))

    @pytest.mark.parametrize("seq_len, chunk_size_test, has_mask_initially", [
        (4, 2, False), (4, 2, True), 
        (5, 2, False), (5, 2, True), 
        (2, 4, False), (2, 4, True), # seq_len < chunk_size (AHCAttention.forward now raises error for this)
                                     # _chunk_input itself should still process it by padding up to chunk_size
    ])
    def test_chunk_input(self, seq_len, chunk_size_test, has_mask_initially):
        batch_size, d_model = 1, 4
        ahc_attention_dummy = self._create_ahc_attention(d_model=d_model, chunk_size=chunk_size_test)

        input_tensor = torch.randn(batch_size, seq_len, d_model)
        
        # This mask is what _process_input_mask would produce
        processed_input_mask_for_chunking = None 
        if has_mask_initially:
            # Simulate a (B, 1, 1, S_orig) float mask as input to _chunk_input
            mask_data = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.float)
            if seq_len > 1:
                mask_data[:, :, :, -1] = 0.0 # Mask last element for testing
            processed_input_mask_for_chunking = mask_data

        padding_expected = 0
        if seq_len % chunk_size_test != 0:
            padding_expected = chunk_size_test - (seq_len % chunk_size_test)
        
        padded_seq_len_expected = seq_len + padding_expected
        num_chunks_expected = padded_seq_len_expected // chunk_size_test
        
        # If seq_len < chunk_size, forward() would error. _chunk_input should pad to 1 chunk.
        if seq_len < chunk_size_test:
            num_chunks_expected = 1 
            padded_seq_len_expected = chunk_size_test
            padding_expected = chunk_size_test - seq_len


        chunked_tensor, chunked_mask, padding_needed, original_seq_len_returned = \
            ahc_attention_dummy._chunk_input(input_tensor, processed_input_mask_for_chunking)

        assert original_seq_len_returned == seq_len
        assert padding_needed == padding_expected
        assert chunked_tensor.shape == (batch_size, num_chunks_expected, chunk_size_test, d_model)

        if has_mask_initially:
            assert chunked_mask is not None
            assert chunked_mask.shape == (batch_size, num_chunks_expected, 1, chunk_size_test)
            
            # Verify mask content after chunking and padding
            # Reconstruct what the chunked mask should look like from processed_input_mask_for_chunking
            if processed_input_mask_for_chunking is not None:
                expected_chunked_mask_content_list = []
                for b in range(batch_size):
                    batch_slice = processed_input_mask_for_chunking[b, 0, 0, :] # Shape (S_orig)
                    if padding_expected > 0:
                        padded_slice = F.pad(batch_slice, (0, padding_expected), mode='constant', value=0.0)
                    else:
                        padded_slice = batch_slice
                    expected_chunked_mask_content_list.append(padded_slice.view(num_chunks_expected, 1, chunk_size_test))
                expected_chunked_mask = torch.stack(expected_chunked_mask_content_list, dim=0)
                assert torch.equal(chunked_mask, expected_chunked_mask)
        else:
            assert chunked_mask is None

class TestAHCAttentionEdgeCases(TestAHCAttentionMasking): # Inherit helper
    def test_seq_len_less_than_chunk_size_error_in_forward(self):
        batch_size, seq_len, d_model = 1, 2, 4
        chunk_size = 4 
        ahc_attention = self._create_ahc_attention(d_model=d_model, chunk_size=chunk_size)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        with pytest.raises(ValueError) as excinfo:
            ahc_attention.forward(query, key, value, input_padding_mask=None)
        assert f"AHCAttention: chunk_size ({chunk_size}) cannot be greater than input sequence length ({seq_len})" in str(excinfo.value)

    def test_config_init_errors(self):
        with pytest.raises(ValueError, match="Invalid summarization_method_config\\['type'\\]"):
            self._create_ahc_attention(summarization_type="INVALID_SUM_TYPE")

        with pytest.raises(ValueError, match="pool_type must be specified for summarization type S1_pool"):
             self._create_ahc_attention(summarization_type="S1_pool", pool_type=None)

        with pytest.raises(ValueError, match="Invalid summarization_method_config\\['pool_type'\\]"):
            self._create_ahc_attention(pool_type="INVALID_POOL_TYPE")
        
        with pytest.raises(ValueError, match="Invalid global_attention_method_config\\['type'\\]"):
            self._create_ahc_attention(global_attention_type="INVALID_GLOBAL_TYPE")

        from my_attention_research_project.models.attention.combination_strategies import get_combination_strategy
        with pytest.raises(ValueError, match="Unsupported combination strategy type"):
            get_combination_strategy({"type": "INVALID_COMB_TYPE"})
        
        with pytest.raises(ValueError, match="Combination strategy type cannot be None"):
            self._create_ahc_attention(combination_type=None) 

        # Test cases that should pass initialization
        try:
            ahc_none_sum = self._create_ahc_attention(summarization_type=None)
            assert ahc_none_sum.summarization_method_config['type'] is None
            # Note: Forward pass with this config will likely fail later due to summarization returning None
            # This test is for __init__ validation only.
            with pytest.raises(ValueError, match="Summarization type not specified"): # Error from forward pass
                q = torch.randn(1,4,4)
                ahc_none_sum.forward(q,q,q)

        except ValueError as e:
            pytest.fail(f"summarization_type=None should be allowed by __init__ config validation, but got: {e}")

        try:
            ahc_none_global = self._create_ahc_attention(global_attention_type=None)
            assert ahc_none_global.global_attention_method_config['type'] is None
            # Forward pass with this config should pass if summarization is valid.
            q = torch.randn(1,4,4)
            output, _ = ahc_none_global.forward(q,q,q) # Should use pass-through for global
            assert output.shape == (1,4,4)

        except ValueError as e:
            pytest.fail(f"global_attention_type=None should be allowed by __init__ config validation, but got: {e}")
