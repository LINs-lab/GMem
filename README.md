# GMem: A Modular Approach for Ultra-Efficient Generative Models

<div align="center">
  Yi Tang :man_student:, <a href="https://sp12138.github.io/">Peng Sun :man_artist:</a>, Zhenglin Cheng :man_student:, <a href="https://tlin-taolin.github.io/">Tao Lin :skier:</a>

  <a href="https://arxiv.org/abs/2412.08781">[arXiv] :page_facing_up:</a> | <a href="#bibliography">[BibTeX] :label:</a>
</div>

![Teaser image](./assets/docs/selected_pics.png)

**ImageNet Generation (w/o cfg or any other guidance techniques):**
- **$256\times 256$**: ~**$20\text{h}$ total training time** ($160$ epochs) → $100$ NFE → **FID $1.53$**  
- **$512\times 512$**: ~**$50\text{h}$ total training time** ($400$ epochs) → $100$ NFE → **FID $1.89$**  



### Abstract

  Recent studies indicate that the denoising process in deep generative diffusion models implicitly learns and memorizes semantic information from the data distribution.
  These findings suggest that capturing more complex data distributions requires larger neural networks, leading to a substantial increase in computational demands, which in turn become the primary bottleneck in both training and inference of diffusion models.
  To this end, we introduce GMem: A Modular Approach for Ultra-Efficient Generative Models.
  Our approach GMem decouples the memory capacity from model and implements it as a separate, immutable memory set that preserves the essential semantic information in the data.
  The results are significant: GMem enhances both training, sampling efficiency, and diversity generation.
  This design on one hand reduces the reliance on network for memorize complex data distribution and thus enhancing both training and sampling efficiency.
  On ImageNet at $256 \times 256$ resolution, GMem achieves a $50\times$ training speedup compared to SiT, reaching **FID $=7.66$** in fewer than $28$ epochs (**$\sim 4$ hours** training time), while SiT requires $1400$ epochs.
  Without classifier-free guidance, GMem achieves state-of-the-art (SoTA) performance **FID $=1.53$** in $160$ epochs with **only $\sim 20$ hours** of training, outperforming LightningDiT which requires $800$ epochs and $\sim 95$ hours to attain FID $=2.17$.

---


### Requirements

- **Python and PyTorch:**
  - 64-bit Python 3.10 or later.
  - PyTorch 2.4.0 or later (earlier versions might work but are not guaranteed).

- **Additional Python Libraries:**
  - A complete list of required libraries is provided in the [requirements.txt](./requirements.txt) file.
  - To install them, execute the following command:
    ```bash
    pip install -r requirements.txt
    ```

---

### Evaluation

To set up the evaluation and sampling of images from the pretrained GMem-XL model, here are the steps to follow:

#### 1. **Download the Pretrained Weights:**

   - **Pretrained model**: Download the pretrained weights for the network and corresponding memory bank from the provided link on Huggingface:

|    Backbone    | Training Epoch | Dataset                   | Bank Size | FID | Download |
|----------------|----------------|---------------------------|-----------|-----|----------|
| LightningDiT-XL|   160          | ImageNet $256\times 256$  | 1.28M     |1.53 | [Huggingface](https://huggingface.co/Tangentone/GMem) |
   
   - **VA-VAE Tokenizer**: You also need the VA-VAE tokenizer. Download the tokenizer from the official repository at [VA-VAE on GitHub](https://github.com/hustvl/LightningDiT/tree/main?tab=readme-ov-file#inference-with-pre-trained-models).

#### 2. **Modify Config Files:**

   - Once you’ve downloaded the necessary pretrained models and tokenizers, modify the following configuration files with the correct paths:
   
     - **For the GMem model (`configs/gmem_sde_xl.yaml`)**:
       - Update the `ckpt_path` with the location where you saved the pretrained weights.
       - Update the `GMem:bank_path` with the location of the bank size data.
       - Also, specify the path to the reference file (`VIRTUAL_imagenet256_labeled.npz`, see [ADM](https://github.com/openai/guided-diffusion) for details) for FID calculation in the `data:fid_reference_file` argument.
       
     - **For the VA-VAE Tokenizer (`tokenizer/configs/vavae_f16d32.yaml`)**:
       - Specify the path to the tokenizer in the `ckpt_path` section of the configuration.

#### 3. **Run Evaluation Scripts:**

   - Use the provided script to sample images and automatically calculate the FID score:
   ```bash
   bash scripts/evaluation_gmem_xl.sh
   ```

---

### Memory Manipulation

#### **External Knowledge Manipulation**

To incorporate external knowledge using previously unseen images, follow the steps below:

1. Store the new images in the `assets/novel_images` directory.
2. Execute the script to generate new images:
   ```bash
   bash scripts/external_knowledge_generation.sh
   ```

#### **Internal Knowledge Manipulation**

To generate new memory snippets by interpolating between two existing images, follow these steps:

1. Place the source images in the `assets/interpolation/lerp/a` and `assets/interpolation/lerp/b` directories, ensuring both images have identical filenames.
2. Run the script to create interpolated images:
   ```bash
   bash scripts/internal_knowledge_generation.sh
   ```

---

### Preparing Data

1. **Set up VA-VAE**: Follow the instructions in the  **Evaluation**  and [VA-VAE tutorial](https://github.com/hustvl/LightningDiT/blob/main/docs/tutorial.md) to properly set up and configure the VA-VAE model. 
   
2. **Extract Latents**: Once VA-VAE is set up, you can run the following script to extract the latents for all ImageNet images:
   ```bash
   bash scripts/preprocessing.sh
   ```
   This script will process all ImageNet images and store their corresponding latents.

3. **Modify the Configuration**: After extracting the latents, you need to update the `data:data_path` in the `configs/gmem_sde_xl.yaml` file. Set this path to the location where the extracted latents are stored. This ensures that GMem-XL can access the processed latents during training.

---

### Train GMem

With the data prepared and the latents extracted, you can proceed to train the GMem-XL model by simply run the following script:

```bash
bash scripts/train_gmem_xl.sh
```


---

### Bibliography

If you find this repository helpful for your project, please consider citing our work:

```bibtex
@article{tang2024generative,
  title={Generative Modeling with Explicit Memory},
  author={Tang, Yi and Sun, Peng and Cheng, Zhenglin and Lin, Tao},
  journal={arXiv preprint arXiv:2412.08781},
  year={2024}
}
```


### Acknowledgement

This code is mainly built upon [VA-VAE](https://github.com/hustvl/LightningDiT), [SiT](https://github.com/willisma/SiT), [edm2](https://github.com/NVlabs/edm2), and [REPA](https://github.com/sihyun-yu/REPA) repositories.


