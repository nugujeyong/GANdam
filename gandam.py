import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Conv2D, LeakyReLU, Lambda ,Flatten, Dense, Reshape, BatchNormalization, Conv2DTranspose, Input, Layer 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import linspace
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
from os import listdir
from copy import deepcopy
from random import randint
from functools import partial
import imgaug.augmenters as iaa

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""
    def call(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGAN():
    def __init__(self):
        self.weight_init = RandomNormal(mean=0., stddev=0.02)
        self.BATCH = 32
        self.W = 128
        self.H = 128
        self.C = 3 
        self.CHECKPOINT = 50
        Datapath = "./gunimdata.npy"
        Datapath2 = "./augimdata.npy"
        self.load_npy(Datapath,Datapath2)

        self.build_discriminator()
        self.build_generator()
        self.build_adversarial()
    
    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)
    
    def _compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var_list)
        #for var, grad in zip(var_list, grads)]
        for grad in grads]
    
    def grad(self, y, x):
        V = Lambda(lambda z: K.gradients(
            z[0], z[1]), output_shape=[1])([y, x])
        return V
    
    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = self.grad(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    def load_npy(self, npy_path1,npy_path2, amount_of_data=0.25):
        self.X_train = np.load(npy_path1)
        self.X_train = self.X_train[:int((amount_of_data)*float(len(self.X_train)))]
        self.X_train=(self.X_train.astype(np.float32) - 127.5)/127.5
        self.X_train=np.expand_dims(self.X_train, axis=3)

        self.Xaug_train = np.load(npy_path2)
        self.Xaug_train = self.Xaug_train[:int((amount_of_data)*float(len(self.Xaug_train)))]
        self.Xaug_train=(self.Xaug_train.astype(np.float32) - 127.5)/127.5
        self.Xaug_train=np.expand_dims(self.Xaug_train, axis=3)

        return 

    def save_npy(self, data_dir, image_shape):
        images = None
        augimages = None
        image_paths = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

        for i, image_path in enumerate(image_paths):
            print("loading_image:{}".format(i))
            try:
                # Load image
                loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)
                #augment image
                aug = iaa.Sequential([ iaa.ChangeColorTemperature((1100, 10000)), iaa.TranslateX(px=(-2, 2)) ])
                # Convert PIL image to numpy ndarray
                loaded_image = np.array(loaded_image)
                # Add another dimension (Add batch dimension)
                loaded_image = np.expand_dims(loaded_image, axis=0)
                augimg = aug.augment_images(loaded_image.astype(np.uint8))
                # Concatenate all images into one tensor
                if images is None:
                    images = loaded_image
                    
                else:
                    images = np.concatenate([images, loaded_image], axis=0)
                
                if augimages is None:
                    augimages = augimg
                    
                else:
                    augimages = np.concatenate([augimages, augimg], axis=0)

            except Exception as e:
                print("Error:", i, e)
        
        np.save('gunimdata.npy',images)
        np.save('augimdata.npy',augimages)
       


    def build_discriminator(self):
        critic_input = Input(shape=(self.H,self.W,3), name='critic_input')
        x = critic_input
        x = Conv2D(filters=64, kernel_size = 5, strides = 2, padding = 'same', name = 'conv_disc_1', kernel_initializer = self.weight_init)(x)
        x =LeakyReLU(alpha = 0.2)(x)
    
        x =Conv2D(filters=128, kernel_size=5, strides = 2, padding='same', name = 'conv_disc_2', kernel_initializer = self.weight_init)(x)
        x=LeakyReLU(alpha = 0.2)(x)
    
        x=Conv2D(filters=256, kernel_size=5, strides = 2, padding='same', name = 'conv_disc_3', kernel_initializer = self.weight_init)(x)
        x=LeakyReLU(alpha = 0.2)(x)
    
        x=Conv2D(filters=512, kernel_size=5, strides = 2, padding='same', name = 'conv_disc_4', kernel_initializer = self.weight_init)(x)
        x=Flatten()(x)
        critic_output=Dense(1, activation = None, kernel_initializer = self.weight_init)(x)
        self.critic = Model(critic_input, critic_output)
        

    def build_generator(self):
        generator_input = Input(shape=(100,), name='generator_input')
       
        x = generator_input
        x =Dense((4*4*512), kernel_initializer = self.weight_init, name = 'input')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x =LeakyReLU(alpha=0.2)(x)
        x = Reshape((4, 4, 512))(x)
        
        x=Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', name= 'conv_gen_1', kernel_initializer = self.weight_init)(x)
        x=BatchNormalization(momentum=0.9)(x)
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2DTranspose(filters=128, kernel_size=5, strides = 2, padding='same', name = 'conv_gen_2', kernel_initializer = self.weight_init)(x)
        x=BatchNormalization(momentum = 0.9)(x)
        x=LeakyReLU(alpha= 0.2)(x)
        x=Conv2DTranspose(filters=64, kernel_size = 5, strides= 2, padding = 'same', name = 'conv_gen_3', kernel_initializer = self.weight_init)(x)
        x=BatchNormalization(momentum=0.9)(x)
        x=LeakyReLU(alpha=0.2)(x)

        x=Conv2DTranspose(filters=64, kernel_size = 5, strides= 2, padding = 'same', name = 'conv_gen_4', kernel_initializer = self.weight_init)(x)
        x=BatchNormalization(momentum=0.9)(x)
        x=LeakyReLU(alpha=0.2)(x)

        x=Conv2DTranspose(filters = 3, kernel_size = 5, strides = 2, padding = 'same', activation = 'tanh', name= 'conv_gen_5', kernel_initializer = self.weight_init)(x)
        generator_output = x
        self.generator = Model(generator_input, generator_output)
        


    def build_adversarial(self):
        self.generator.trainable = False

        real_img = Input(shape=(self.H,self.W,3))
        z_disc = Input(shape = (100,))
        fake_img = self.generator(z_disc)
        
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        interpolated_img = RandomWeightedAverage(self.BATCH)([real_img, fake_img])

        validity_interpolated = self.critic(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss,
                                    interpolated_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs = [real_img, z_disc],
                                outputs = [valid,fake,validity_interpolated])

        self.critic_model.compile(
            loss = [ self.wasserstein, self.wasserstein, partial_gp_loss]
            ,optimizer = Adam(learning_rate = 0.0002)
            , loss_weights = [1, 1, 10]
        )

        self.critic.trainable = False
        self.generator.trainable = True

        model_input = Input(shape=(100,))
        img = self.generator(model_input)
        model_output = self.critic(img)
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=Adam(learning_rate=0.0002),
                            loss = self.wasserstein)
        self.critic.trainable =True


    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(0, 1, (batch_size, 100))
        return self.model.train_on_batch(noise, valid)


    def train(self, batch_size, epochs):

        epoch_list=[]
        dis_real = []
        dis_fake = []
        dis_intpol = []
        gen_list = []
        g_p = []
        directory = './weight'
        if not os.path.exists(directory):
            os.mkdir(directory)
            print("Directory folder Created ")
        for epoch in range(epochs):
            b=0
            disc_loss = []
            gen_loss = []

            X_train_temp=deepcopy(self.X_train)
            X_aug_temp = deepcopy(self.Xaug_train)
            while len(self.X_train)-b*self.BATCH > self.BATCH:
                b = b+1
                
                count_real_images = int(batch_size/2)
                starting_index = randint(0, (len(X_train_temp)-count_real_images))
                starting_index2 = randint(0, (len(X_aug_temp)-count_real_images))
                real_images_raw = X_train_temp[starting_index : (starting_index 
                                                            + count_real_images)]
                aug_images_raw = X_aug_temp[starting_index2 : (starting_index2 
                                                             + count_real_images)]
                
                X_train_temp = np.delete(X_train_temp, range(starting_index,(starting_index + count_real_images)), 0)
                X_aug_temp = np.delete(X_aug_temp, range(starting_index2,(starting_index2 + count_real_images)), 0)
                x_aug_batch = aug_images_raw.reshape(count_real_images,
                                             self.W, self.H, self.C)
                x_batch = real_images_raw.reshape(count_real_images,
                                            self.W, self.H, self.C)
                x_batch = np.concatenate([x_batch,x_aug_batch],axis=0)
               
                noise = np.random.normal(0, 1, (batch_size, 100))
                for _ in range(5):
                    valid = np.ones((batch_size,1), dtype=np.float32)
                    fake = -np.ones((batch_size,1), dtype=np.float32)
                    dummy = np.zeros((batch_size, 1), dtype=np.float32)
                    d_loss = self.critic_model.train_on_batch([x_batch, noise], [valid, fake, dummy])

                g_loss = self.train_generator(batch_size)

                disc_loss.append(d_loss)
                gen_loss.append(g_loss)
            
            
            dis_sum = np.sum(disc_loss, 0)
            gen_loss = sum(gen_loss)/len(gen_loss)
                
            
            dis_real.append(dis_sum[0]/len(dis_sum))
            dis_fake.append(dis_sum[1]/len(dis_sum))
            dis_intpol.append(dis_sum[2]/len(dis_sum))
            gen_list.append(gen_loss)   
            g_p.append(dis_sum[3]/len(dis_sum))     
            epoch_list.append(epoch)
            
            print("Epoch: ",epoch,"[D loss: ", dis_sum/len(dis_sum),"], [G loss: ", gen_loss,"]")

            plt.subplot(2,1,1)
            plt.plot(epoch_list,dis_real,color = 'green')
            plt.plot(epoch_list,dis_fake,color = 'red')
            plt.plot(epoch_list,dis_intpol,color = 'black')
            plt.plot(epoch_list,gen_list,color = 'yellow')
            plt.title("Wasserstein Loss")
            plt.legend(["dis_real","dis_fake","dis_intp","gen"])
           
            
            plt.subplot(2,1,2)
            plt.plot(epoch_list,g_p,color = 'blue')
            plt.title("Gradient Penalty")
            plt.legend(["gradient_penalty"])
            plt.tight_layout()
            plt.savefig('wasslossplot128.png')

            if epoch % self.CHECKPOINT == 0:
                label = str(epoch)
                self.plot_checkpoint(label)
                self.generator.save_weights("./weight/generator_epoch_{}.h5".format(epoch))
                
    
        # Save networks
        try:
            self.generator.save_weights("./weight/generator_epoch_{}.h5".format(epoch))
            
        except Exception as e:
            print("Error:", e)
        return


    def plot_checkpoint(self,e):
            directory = './results'
            if not os.path.exists(directory):
                os.mkdir(directory)
                print("Directory folder Created ")
            filename = "./results/sample_"+str(e)+".png"
            
            noise = np.random.normal(0, 1, (16, 100))
            images = self.generator.predict(noise)
            
            plt.figure(figsize=(10,10))
            for i in range(images.shape[0]):
                plt.subplot(4, 4, i+1)
                if self.C==1:
                    image = images[i, :, :]
                    image = np.reshape(image, [self.H,self.W])
                    image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                    plt.imshow(image,cmap='gray')
                elif self.C==3:
                    
                    image = images[i, :, :, :]
                    image = np.reshape(image, [self.H,self.W,self.C])
                    
                    image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                    plt.imshow(image)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close('all')
            return
    

# uniform interpolation between two points in latent space
    def interpolate_points(self,p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            vectors.append(v)
        return np.asarray(vectors)

   

    def test(self,epoch):
        self.generator.load_weights("./weight/generator_epoch_{}.h5".format(epoch))
        noise = np.random.normal(0, 1, (36, 100))
        images = self.generator.predict(noise)
        
                

        filename = "36image.png"
        plt.figure(figsize=(20,20))
        for i in range(images.shape[0]):
            plt.subplot(6, 6, i+1)
            if self.C==1:
                image = images[i, :, :]
                image = np.reshape(image, [self.H,self.W])
                image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                plt.imshow(image,cmap='gray')
            elif self.C==3:
                
                image = images[i, :, :, :]
                image = np.reshape(image, [self.H,self.W,self.C])
                
                image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        

WGAN().train(32, 100000)
#WGAN().test(10000)