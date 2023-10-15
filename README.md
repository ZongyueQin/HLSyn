# Towards a Comprehensive Benchmark for FPGA Targeted High-Level Synthesis

## Description

High-level synthesis (HLS) aims to raise the abstraction layer in hardware design, enabling the design of domain-specific accelerators (DSAs) like field-programmable gate arrays (FPGAs) using C/C++ instead of hardware description languages (HDLs). Compiler directives in the form of pragmas play a crucial role in modifying the microarchitecture within the HLS framework. However, the space of possible microarchitectures grows exponentially with the number of pragmas. Moreover, the evaluation of each candidate design using the HLS tool consumes significant time, ranging from minutes to hours, leading to a time-consuming optimization process. To accelerate this process, machine learning models have been used to predict design quality in milliseconds. However, existing open-source datasets for training such models are limited in terms of design complexity and available optimizations. In this paper, we present HLSyn, the first benchmark that addresses these limitations. It contains more complex programs with a wider range of optimization pragmas, making it a comprehensive dataset for training and evaluating design quality prediction models. The HLSyn benchmark consists of 42 unique programs/kernels, resulting in over 42,000 labeled designs. We conduct an extensive comparison of state-of-the-art baselines to assess their effectiveness in predicting design quality. As an ongoing project, we anticipate expanding the HLSyn benchmark in terms of both quantity and variety of programs to further support the development of this field.

## Citation

If you use the dataset in your work, please cite our paper.
```
@article{chang2023dr,
  title={Towards a Comprehensive Benchmark for FPGA Targeted High-Level Synthesis},
  author={Yunsheng Bai, Atefeh Sohrabizadeh, Zongyue Qin, Ziniu Hu, Yizhou Sun, and Jason Cong},
  journal={NeurIPS},
  year={2023}
}
```

## Download Link

[Temporary Link](https://zenodo.org/records/8034115) 

## Data Content and Format

TODO

## Leaderboard

Coming soon

## Running Baselines

Check instructions [here](https://github.com/ZongyueQin/HLSyn/tree/main/baselines) 

TODO: Current baselines run with the entire test set, need to address data leak problem 

## Acknowledgement

TODO
