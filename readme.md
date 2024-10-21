<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[stars-shield]: https://img.shields.io/github/stars/TheLuoFengLab/ChemFM.svg?style=flat-square&color=b75347
[stars-url]: https://github.com/TheLuoFengLab/ChemFM/stargazers
[forks-shield]: https://img.shields.io/github/forks/TheLuoFengLab/ChemFM.svg?style=flat-square&color=df7e66
[forks-url]: https://github.com/TheLuoFengLab/ChemFM/network/members
[issues-shield]: https://img.shields.io/github/issues/TheLuoFengLab/ChemFM.svg?style=flat-square&color=edc775
[issues-url]: https://github.com/TheLuoFengLab/ChemFM/issues
[license-shield]: https://img.shields.io/github/license/TheLuoFengLab/ChemFM.svg?style=flat-square&color=94b594
[license-url]: https://github.com/othneildrew/TheLuoFengLab/ChemFM/blob/master/LICENSE.txt


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">ChemFM: A Foundation Model for Chemistry</h1>

  [![Stargazers][stars-shield]][stars-url]
  [![Forks][forks-shield]][forks-url]
  [![Issues][issues-shield]][issues-url]
  [![MIT License][license-shield]][license-url]

  <p align="center">
    <a href="https://huggingface.co/ChemFM">
      <img src="https://info.arxiv.org/brand/images/brand-supergraphic.jpg" alt="arxiv" width="25" height="25" style="vertical-align: middle; margin-right: 0px;">
    </a>    
    <a href="https://huggingface.co/ChemFM">
      ArXiv
    </a>
    |
    <a href="https://huggingface.co/ChemFM">
      <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20" height="20" style="vertical-align: middle; margin-right: 0px;">
    </a>    
    <a href="https://huggingface.co/ChemFM">
      Hugging Face
    </a>
    |
    <a href="https://discord.gg/xjyVaZ9V">
      <img src="https://camo.githubusercontent.com/ae76bfbcd3ea4af324682842213b28d9a7ebdd8791d8531d1b7e3b8b4d2a0302/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f646973636f72642e737667" alt="Discord" width="25" height="25" style="vertical-align: middle; margin-right: 0px;">
    </a>    
    <a href="https://discord.gg/xjyVaZ9V">
      Discord
    </a>
  </p>

  <p align="center">
    <a href="https://github.com/TheLuoFengLab/ChemFM/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/TheLuoFengLab/ChemFM/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

ChemFM is a large-scale foundation model (with 1B and 3B parameters) specifically developed for chemistry, pre-trained on 178 million molecules from [UniChem](https://www.ebi.ac.uk/unichem/) using self-supervised causal language modeling to extract generalizable molecular representations. 

<p align="center">
  <img src="images/pretrain.jpg" alt="Pretraining Overview" width="800">
</p>

This model can be adapted to diverse downstream chemical applications including
* Molecular property prediction
* Conditional molecular generation 
* Reaction synthesis and retro-synthesis predictions. 
* and so on ...

<p align="center">
  <img src="images/finetune.jpg" alt="Pretraining Overview" width="800">
</p>



<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
The ChemFM is tested on Python 3.10 and PyTorch 2.4.1. The environment can be installed easily via a conda environment according to the following steps:

* Clone the Repository
  ```bash
  git clone https://github.com/TheLuoFengLab/ChemFM.git
  cd ChemFM
  ```
* Create and Activate Conda Environment
  ```bash
  conda env create -f environment.yml 
  conda activate ChemFM
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] [Pre-training]()
- [ ] [Molecular property prediction]()
- [ ] [Conditional molecular generation]()
- [ ] [Reaction prediction]()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Main developer: Feiyang Cai - feiyang@clemson.edu

Project superviser: Feng Luo - luofeng@clemson.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation
If you find our work valuable, please cite:
```
@article{ChemFM,
       author = {Cai, Feiyang and Luo, Feng},
        title = {ChemFM: A foundation model for Chemisty},
      journal = {arXiv preprint arXiv:2203.08441},
         year = 2024,
}
```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [TinyLLama](https://github.com/jzhang38/TinyLlama)
* [Hugging Face](https://huggingface.co/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
