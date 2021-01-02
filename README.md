# GANdam
'GANdam' generates new gundam images with GAN(Generative Adversarial Networks).

With 'GANdam', you can generate numerous Mobile Suit designs that draws back to your childhood memories.

[https://github.com/nugujeyong/GANdam](https://github.com/nugujeyong/GANdam)
## Results
![36image](https://user-images.githubusercontent.com/59949284/103440547-e7e97780-4c89-11eb-831a-55fa35f50024.png)
![sample14](https://user-images.githubusercontent.com/59949284/103458054-c4850200-4d47-11eb-8e5b-4c8d13cc3423.png)
![sample30](https://user-images.githubusercontent.com/59949284/103458055-c64ec580-4d47-11eb-980f-dd6bed7bea98.png)
![sample80](https://user-images.githubusercontent.com/59949284/103458057-c77ff280-4d47-11eb-8980-dc22f2775a6a.png)
## Walking through the latent space
![sample1](https://user-images.githubusercontent.com/59949284/103439926-a60a0280-4c84-11eb-8b5e-a14aa55ae2a8.gif)

![sample2](https://user-images.githubusercontent.com/59949284/103440005-74456b80-4c85-11eb-86f1-a929a65d5e82.gif)

![sample4](https://user-images.githubusercontent.com/59949284/103440274-5d077d80-4c87-11eb-94bc-033dd11dc030.gif)


## Dataset
Images from [Gundam Wiki](https://gundam.fandom.com/wiki/The_Gundam_Wiki)

Total 1241 images were used for training.

## Training
```
cd "folder_to_the_project"
python save_npy.py
python gandam.py
```


