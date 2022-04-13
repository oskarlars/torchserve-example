# torchserve-example

Example TorchServe handler that takes an image input in the request body and serves a JPEG image as response. 

## Archiving

Run this command to archive the model:

```torch-model-archiver --model-name tinymodel --version 1.0 --model-file model.py --export-path model_store --handler handler.py -f```

This will save the model archive into the model_store folder (you might have to create this directory first).

## Testing

Run this command to start and test the model locally.

```torchserve --start --ncs --model-store model_store --models tinymodel.mar```

To stop the server, run the following command.

```torchserve --stop```

To test the http endpoint with an input image (kitten.jpg) and output image (out.jpg), run the following command

```http POST http://127.0.0.1:8080/predictions/tinymodel/ @kitten.jpg > out.jpg```

This uses [HTTPie](https://httpie.io/)
