dependencies = ['torch', 'torchaudio']

def supervoice():

    # Imports
    import torch
    import os
    from supervoice_valle import SupervoceNARModel, SupervoceARModel, Tokenizer, Supervoice

    # Load tokenizer
    tokenizer = Tokenizer(os.path.join(os.path.dirname(__file__), "tokenizer_text.model"))

    # Load encodec
    vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)

    # Load checkpoints
    ar_model = SupervoceARModel()
    nar_model = SupervoceNARModel()
    checkpoint_ar = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice-valle-ar-600000.pt", map_location="cpu")
    checkpoint_nar = torch.hub.load_state_dict_from_url("https://shared.korshakov.com/models/supervoice-valle-nar-600000.pt", map_location="cpu")
    ar_model.load_state_dict(checkpoint_ar['model'])
    nar_model.load_state_dict(checkpoint_nar['model'])

    # Create model
    model = Supervoice(ar_model, nar_model, encodec_model, vocos, tokenizer)

    # Switch to eval mode
    model.eval()

    return model
            
