from . import make_model, train, eval_model

def pipeline(model_type, config, data_bundle, pretrained_models=None):
    train_df, val_df, test_df, train_loader, val_loader, _, _, n_users, n_movies, min_rating, max_rating = data_bundle

    model = make_model(model_type, config, n_users, n_movies, pretrained_models)

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        name=f"best_{model_type}",
        device=config['device'],
        min_rating=min_rating,
        max_rating=max_rating
    )

    eval_model(model, test_df, config['batch_size'], config['device'], min_rating, max_rating)

    return model