from transformers import EncoderDecoderConfig


class TransformerVaeConfig(EncoderDecoderConfig):
    model_type = "vae"
    is_composition = True

    def __init__(
        self,

        latent_size=32,
        share_embeddings=True,

        vae_encoder_n_layers=1,
        vae_encoder_use_dropout=False,

        vae_decoder_n_layers=1,
        vae_decoder_use_dropout=False,
        vae_decoder_use_layer_norm=False,
        vae_decoder_layer_norm_eps=1e-09,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_input = self.encoder.d_model
        self.d_output = self.decoder.d_model
        self.latent_size = latent_size
        self.share_embeddings = share_embeddings

        self.vae_encoder_n_layers = vae_encoder_n_layers
        self.vae_encoder_use_dropout = vae_encoder_use_dropout

        self.vae_decoder_n_layers = vae_decoder_n_layers
        self.vae_decoder_use_dropout = vae_decoder_use_dropout
        self.vae_decoder_use_layer_norm = vae_decoder_use_layer_norm
        self.vae_decoder_layer_norm_eps = vae_decoder_layer_norm_eps

        self.is_encoder_decoder = True

