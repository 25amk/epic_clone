"""
inference_test.py - Test the inference pipeline for job prediction
"""
import json
import pytest
from .regression import jobstat_inference, extract_relevant_features_from_query, create_job_pred_chain, create_job_pred_chain_as_tool


@pytest.mark.integration
def test_jobstat_inference():
    output_feature = ['stats_total_node_energy','stats_total_gpu_energy']
    user_input = {
        'domain': ['STF', 'CFD'], 
        'node_count': [1024, 2048], 
        'time_elapsed': [7200], 
        'utilization_type': ['cpu', 'gpu'] 
    }

    result = jobstat_inference(user_input, output_feature)

    # See if we did get something from the inference
    assert result is not []
    assert len(result) > 0

    # Test the shape of the first element
    for entry in result:
        assert "domain" in entry
        assert "node_count" in entry
        assert "time_elapsed" in entry
        assert "utilization_type" in entry
        assert "prediction" in entry

        # All output features should be found in the predictions
        for feature in output_feature:
            assert feature in entry["prediction"]

        # All predictions should be in what we have asked for
        for feature in entry["prediction"].keys():
            assert feature in output_feature


@pytest.mark.integration
def test_run_inference_on_question_base_good_input(small_model):
    """Test the natural language -> input features pipeline"""
    question = """What would be the average temperature range for a molecular dynamics simulation running on GPU on 8 nodes for 180 minutes?"""
    result = extract_relevant_features_from_query(question, model=small_model)

    # Simply test that we have a legitimate dictionary
    assert result is not None
    assert "X" in result
    assert "Y" in result
    assert type(result["X"]) == dict
    assert type(result["Y"]) == list
    assert json.dumps(result)


@pytest.mark.integration
def test_run_inference_on_question_base_bad_input(small_model):
    """Test the natural language -> input features pipeline"""
    question = """Who are you?"""
    result = extract_relevant_features_from_query(question, model=small_model)
    assert result == None


@pytest.mark.integration
def test_chain(small_model):
    chain = create_job_pred_chain(small_model)
    question = """What would be the average temperature range of the GPUs for a molecular dynamics simulation running on GPU on 8 nodes for 180 minutes?"""
    result = chain.invoke(question)
    print(result)


@pytest.mark.integration
def test_tool_call(large_model):
    """Test whether the chain as the tool is actually requested"""

    # The tool itself isn't called, so we just pass on any model here
    tool = create_job_pred_chain_as_tool(large_model)

    # Bind the tool with the large model
    llm = large_model.bind_tools([tool])

    # Process a tool invoking query
    question = """Predict the average temperature range of the GPUs for a molecular dynamics simulation running on GPU on 8 nodes for 180 minutes?"""
    result = llm.invoke(question)

    # See if we do have a tool call from the model
    assert len(result.tool_calls) > 0
    assert result.tool_calls[0]["name"] == 'job_pred'
    assert result.tool_calls[0]["args"]["question"] != ""
    for keyword in ["temperature", "molecular"]:
        assert keyword in result.tool_calls[0]["args"]["question"] 

