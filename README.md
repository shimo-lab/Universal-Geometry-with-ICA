# Universal-Geometry-with-ICA

> [Discovering Universal Geometry in Embeddings with ICA](https://aclanthology.org/2023.emnlp-main.283/)                 
> [Hiroaki Yamagiwa](https://ymgw55.github.io/)\*, [Momose Oyama](https://momoseoyama.github.io/)\*, [Hidetoshi Shimodaira](http://stat.sys.i.kyoto-u.ac.jp/members/shimo/)                 
> *EMNLP 2023*

## English word embeddings

### Heatmap of ICA-transformed word embeddings

<img src="images/heatmap_geom31.png" alt="heatmap" width="70%">


## Cross-lingual embeddings

### Heatmaps of ICA-transformed word embeddings

<img src="images/en-es-ru-ar-hi-zh-ja-100-2row-ica-150dpi.png" alt="cross-lingual heatmap">

### Spiky shape of embedding distributions

<img src="images/ica_shape.png" alt="ica shape" width="70%">

### Scatter plots of  ICA-transformed word embeddings

<table>
  <tr>
    <th style="width: 50%;">English</th>
    <th style="width: 50%;">Spanish</th>
  </tr>
  <tr>
    <td><img src="images/en_normed_proj.png" alt="ica en"></td>
    <td><img src="images/es_normed_proj.png" alt="ica es"></td>
  </tr>
</table>

<table>
  <tr>
    <th style="width: 20%;">Russian</th>
    <th style="width: 20%;">Arabic</th>
    <th style="width: 20%;">Hindi</th>
    <th style="width: 20%;">Chinese</th>
    <th style="width: 20%;">Japanese</th>
  </tr>
  <tr>
    <td><img src="images/ru_normed_proj.png" alt="ica ru"></td>
    <td><img src="images/ar_normed_proj.png" alt="ica ar"></td>
    <td><img src="images/hi_normed_proj.png" alt="ica hi"></td>
    <td><img src="images/zh_normed_proj.png" alt="ica zh"></td>
    <td><img src="images/ja_normed_proj.png" alt="ica ja"></td>
  </tr>
</table>

## Code and Data

- The code for English embeddings is currently being prepared.
- For cross-lingual embeddings, dynamic embeddings, and image model embeddings, please refer to the [universal](universal/) directory.


## Citation
If you find our code or data useful in your research, please cite our paper:
```
@inproceedings{DBLP:conf/emnlp/YamagiwaOS23,
  author       = {Hiroaki Yamagiwa and
                  Momose Oyama and
                  Hidetoshi Shimodaira},
  editor       = {Houda Bouamor and
                  Juan Pino and
                  Kalika Bali},
  title        = {Discovering Universal Geometry in Embeddings with {ICA}},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023, Singapore, December 6-10, 2023},
  pages        = {4647--4675},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.emnlp-main.283},
  timestamp    = {Wed, 13 Dec 2023 17:20:20 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/YamagiwaOS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```