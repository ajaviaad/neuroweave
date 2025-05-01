Neuroweave is a derivative model originally based on the Mistral-7B architecture. While it inherits structural elements from the base Mistral model, Neuroweave introduces substantial architectural modifications, including:

Custom activation functions (AutoMixedActivation)

Non-standard MLP activation fusion

Forward computation changes affecting gradient flow and optimization

Structural patching across multiple transformer blocks

Non-bitwise-identical weights due to post-training modification

As a result, the Neuroweave model does not retain the original weight values or behaviors of Mistral and is considered an independently evolved model.

## Original base model:

Mistral-7B (by Mistral AI), licensed under the Apache License 2.0

## Modifications and derivative work:

Neuroweave (by Muhammad Adeel Javaid), licensed under custom license with clear attribution of modifications.

This model no longer distributes Mistral's original weights, but modified weights generated via alternate computation pathways.
