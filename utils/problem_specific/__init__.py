"""
Problem-specific utilities loader.

Dynamically imports problem-specific visualization and evaluation utilities.
"""


def get_visualization_module(problem_name: str):
    """
    Dynamically import visualization module for the problem.
    
    Args:
        problem_name: Name of the problem (e.g., 'schrodinger', 'wave1d', 'burgers1d', 'burgers2d')
        
    Returns:
        Tuple of visualization functions:
        (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
         visualize_ncc_classification, visualize_ncc_classification_input_space,
         visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap,
         visualize_ncc_classification_accuracy_changes, 
         visualize_ncc_classification_input_space_accuracy_changes)
        
    Raises:
        ValueError: If problem name is unknown
    """
    if problem_name == 'schrodinger':
        from .schrodinger_viz import (
            visualize_dataset, visualize_evaluation, 
            visualize_ncc_dataset, visualize_ncc_classification,
            visualize_ncc_classification_input_space,
            visualize_ncc_classification_heatmap,
            visualize_ncc_classification_input_space_heatmap,
            visualize_ncc_classification_accuracy_changes,
            visualize_ncc_classification_input_space_accuracy_changes
        )
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap,
                visualize_ncc_classification_accuracy_changes, 
                visualize_ncc_classification_input_space_accuracy_changes)
    elif problem_name == 'wave1d':
        from .wave1d_viz import (
            visualize_dataset, visualize_evaluation, 
            visualize_ncc_dataset, visualize_ncc_classification,
            visualize_ncc_classification_input_space,
            visualize_ncc_classification_heatmap,
            visualize_ncc_classification_input_space_heatmap,
            visualize_ncc_classification_accuracy_changes,
            visualize_ncc_classification_input_space_accuracy_changes
        )
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap,
                visualize_ncc_classification_accuracy_changes, 
                visualize_ncc_classification_input_space_accuracy_changes)
    elif problem_name == 'burgers1d':
        from .burgers1d_viz import (
            visualize_dataset, visualize_evaluation, 
            visualize_ncc_dataset, visualize_ncc_classification,
            visualize_ncc_classification_input_space,
            visualize_ncc_classification_heatmap,
            visualize_ncc_classification_input_space_heatmap,
            visualize_ncc_classification_accuracy_changes,
            visualize_ncc_classification_input_space_accuracy_changes
        )
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap,
                visualize_ncc_classification_accuracy_changes, 
                visualize_ncc_classification_input_space_accuracy_changes)
    elif problem_name == 'burgers2d':
        from .burgers2d_viz import (
            visualize_dataset, visualize_evaluation, 
            visualize_ncc_dataset, visualize_ncc_classification,
            visualize_ncc_classification_input_space,
            visualize_ncc_classification_heatmap,
            visualize_ncc_classification_input_space_heatmap,
            visualize_ncc_classification_accuracy_changes,
            visualize_ncc_classification_input_space_accuracy_changes
        )
        return (visualize_dataset, visualize_evaluation, visualize_ncc_dataset, 
                visualize_ncc_classification, visualize_ncc_classification_input_space,
                visualize_ncc_classification_heatmap, visualize_ncc_classification_input_space_heatmap,
                visualize_ncc_classification_accuracy_changes, 
                visualize_ncc_classification_input_space_accuracy_changes)
    else:
        raise ValueError(f"Unknown problem: {problem_name}")

