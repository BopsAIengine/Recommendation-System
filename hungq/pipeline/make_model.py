from hungq.models import GMF, AttentionNet, NeuMF, LightGCN, LightGCNPP, SimGCL, NCF, DMF, Ensemble
from hungq.util import build_edge_index

def make_model(model_type, config, data_bundle, n_users, n_movies, pretrained_models=None):
    device = config['device']
    global_mean = data_bundle[0]['rating'].mean()

    if model_type == 'gmf':
        model = GMF(n_users, n_movies, config['embedding_dim'], global_mean=global_mean)

    elif model_type == 'attention':
        model = AttentionNet(
            n_users=n_users,
            n_movies=n_movies,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            n_attention_blocks=config['n_attention_blocks'],
            num_heads=config['n_heads'],
            dropout=config['dropout'],
            global_mean=global_mean
        )
        
    elif model_type == 'dmf':
        model = DMF(
            n_users=n_users,
            n_items=n_movies,
            embedding_dim=config['embedding_dim'],
            user_hidden_dims=config.get('user_hidden_dims', [128, 64]),
            item_hidden_dims=config.get('item_hidden_dims', [128, 64]),
            dropout=config.get('dropout', 0.1),
            global_mean=global_mean
        )
        
    elif model_type == 'ncf':
        if not pretrained_models or 'gmf' not in pretrained_models or 'attention' not in pretrained_models:
            raise ValueError("NCF requires pretrained 'gmf' and 'attention' models passed in 'pretrained_models'")

        gmf_pretrained = pretrained_models['gmf']
        attn_pretrained = pretrained_models['attention']

        model = NCF(gmf_pretrained, attn_pretrained, dropout=config.get('dropout', 0.1))

        if config.get('freeze_pretrained', False):
            for param in model.gmf.parameters():
                param.requires_grad = False
            for param in model.attn_net.parameters():
                param.requires_grad = False
                
    elif model_type == 'nmf':
        model = NeuMF(
            n_users=n_users,
            n_items=n_movies,
            embedding_dim=config['embedding_dim'],
            mlp_hidden_dims=config.get('mlp_hidden_dims', [128, 64]),
            dropout=config.get('dropout', 0.1),
            global_mean=global_mean
        )

    elif model_type == 'lightgcn':
        edge_index = build_edge_index(config['train_df']).to(device)
    
        model = LightGCN(
            n_users=n_users,
            n_items=n_movies,
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            edge_index=edge_index,
            global_mean=global_mean
        )
        
    elif model_type == 'lightgcnpp':
        edge_index = build_edge_index(config['train_df']).to(device)
    
        model = LightGCNPP(
            n_users=n_users,
            n_items=n_movies,
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            edge_index=edge_index,
            global_mean=global_mean,
            residual=True
        )

    elif model_type == 'simgcl':
        edge_index = build_edge_index(config['train_df']).to(device)
    
        model = SimGCL(
            n_users=n_users,
            n_items=n_movies,
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            edge_index=edge_index,
            global_mean=global_mean,
            eps=config.get('eps', 0.1),
            temperature=config.get('temperature', 0.2),
            lambda_cl=config.get('lambda_cl', 0.1)
        )

    elif model_type == 'ensemble':
        if not pretrained_models:
            raise ValueError("Ensemble requires pretrained_models")
    
        model_list = []
        for name in config['ensemble_models']:
            model_list.append(pretrained_models[name])
    
        model = Ensemble(
            models=model_list,
            learn_weights=config.get('learn_weights', True)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)