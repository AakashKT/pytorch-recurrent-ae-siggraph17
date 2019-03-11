# Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder , PyTorch implementation
Link to original paper (SIGGRAPH '17) : https://research.nvidia.com/publication/interactive-reconstruction-monte-carlo-image-sequences-using-recurrent-denoising

This is the unofficial PyTorch implementation of the above paper. 

# Dataset Preparation
The input to this network is :
* Albedo demodulated 1spp RGB image
  * Render a scene using any standart path tracer, and divide it by the albedo image of the same scene. Albedo image is usually obtained by adding the 'DiffuseColor' and 'GlossyColor' passes from the renderer.
* Depth (Z-buffer)
* Normal Map
* Material roughness Map
  * Add the 'GlossyDirect' and 'GlossyIndirect' components from the renderer. Take the inverse of this image.
  
The output of this network is :
* Albedo demodulated 250spp RGB image

Construct the input as one image, as follows : <br>
<img src="https://github.com/AakashKT/pytorch-recurrent-ae-siggraph17/blob/master/00091.png?raw=true" width="256" height="384">
<br>
<table>
  <tr>
    <th> Column 1 </th>
    <th> Column 2 </th>
  </tr>
  <tr>
    <td>1 spp input</td>
    <td>250 spp output</td>
  </tr>
  <tr>
    <td>Albedo Image</td>
    <td>Normal Map</td>
  </tr>
  <tr>
    <td>Depth Map</td>
    <td>Roughness Map</td>
  </tr>
</table>
<br>
<h3> Directory structure </h3>
Make batches of batch size 7 of continuous frames, and put them in a directory (seq_0, seq_1 ..). Do this for all frames.
Split the resulting directories into test and train.

<b> Note : </b> The data directory must contain 'train' and 'test' directories, and these directories much contain directories where the sequence is stored.

* [DATA_DIRECTORY]/train
  * [DATA_DIRECTORY]/train/seq_0/
  * [DATA_DIRECTORY]/train/seq_1/
  * ....
* [DATA_DIRECTORY]/test
  * [DATA_DIRECTORY]/test/seq_0/
  * [DATA_DIRECTORY]/test/seq_1/
  * ....

# Training the network
To train the network, run the following command :

```
  python train.py --data_dir [PATH_TO_DATA_DIRECTORY] --name [EXP_NAME] --save_dir [PATH_TO_SAVE_CHECKPOINTS] --epochs 500
```

# Running a trained network
To test the network, run the following command :

```
  python test.py --data_dir [PATH_TO_DATA_DIRECTORY] --output_dir [PATH_TO_SAVE_RESULTS] --checkpoint [PATH_TO_CHECKPOINT].pt
```

# TODO / Possible faults
* Can we get the demodulated albedo directly from the renderer?
* The network denoises the demodulated albedo perfectly. But while reconstructing the textured image, by multiplication with albedo, it looses a lot of detail. FIX THIS.
