## Future work

### Meta-model training
Solving these bussiness challenge is not just training the model, but understanding the overall product.
In this sense, there are bussiness (product) decisions that would affect the behaviour of the model, and therefore we should change some training or hyperparameter decisions.

Some questions to ask are:
- ***Should we recommend 'standard' prices, or succesful prices?***: By filtering out data of properties that have not been sold, and also spam / other ads, we could give a better recommendation of the price to stipulate in the ad. Moreover, we could even make more advanced models, trying to predict when the property would be sold, for different prices.
- ***When would the customer get a recommendation on the price?***: If the recommendation is given in batch (whenever they click some button) is not the same as if we give some alerts whenever a posting of theirs is not getting attention / needs to be updated. This decision could change temporal features (created_at_first, updated_at)
- ***When would this model be re-trained, and with which periodicity***: this could affect the decision on which temporal features to use (features gotten from created_at_first, updated_at). For example, if training quite often, we could standardize the updated_at feature often. But, if not, we would be inferring data for an out-of-distribution feature.
- ***Do we want an explainable model?***: If we want to be 

### Feature transformation
- When extracting features from the title and description, we removed numeric characters. The presence of these can indeed change the value of the properties. Studying the behaviour with respect to the presence of numbers, and generating features from this, could be interesting.
- One could use pre-trained Word2Vec models for extracting features from title and description, such as the ones in: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/. However, for time constraints, finding a good, lightweight model is not trivial.