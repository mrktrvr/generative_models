# Factor Analysis

## Overview

$\boldsymbol{Y}$: Observation

$\boldsymbol{Z}$: Latent variables

$\boldsymbol{\Lambda}$: Factor loading 

$\boldsymbol{R}$: Uniqueness



## Model

$$
\begin{align}
p(\boldsymbol{Y},\boldsymbol{Z},\boldsymbol{\Lambda},\boldsymbol{R},\boldsymbol{\pi})&=
p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{\Lambda},\boldsymbol{R})p(\boldsymbol{Z})p(\boldsymbol{\Lambda})p(\boldsymbol{R})
\\
p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{\Lambda},\boldsymbol{R})
&=\prod_{t=1}^T\prod_{d=1}^D\mathcal{N}(\boldsymbol{y}_{d,t}|\boldsymbol{\Lambda}_d\boldsymbol{z}_t,\boldsymbol{R}_d)
\\
p(\boldsymbol{\Lambda})&=\prod_{d=1}^{D}\mathcal{N}(\boldsymbol{\Lambda}_d|\boldsymbol{\mu}_{\boldsymbol{\Lambda}_d},\boldsymbol{\Sigma}_{\boldsymbol{\Lambda}_d})
\\
p(\boldsymbol{R})
&=\prod_{d=1}^{D}\mathrm{Gam}(\boldsymbol{R}_{d}|a_{R_d},b_{R_d})
\\
p(\boldsymbol{Z})&=\prod_t^T \mathcal{N}(z_{t}|\mu_{z},\Sigma_{z})=\prod_t^T \prod_{l=1}^{L + 1} \mathcal{N} (z_{l,t}|\mu_{z_l},\sigma_{z_l}^2)
\\
&\mu_{z_l}=\left\{\begin{matrix}
0,&l&=& 1,\cdots,L
\\
1,&l&=&L + 1
\end{matrix}\right.
\\
&\sigma_{z_l}^{2}=\left\{\begin{matrix}
1,&l&=&1,\cdots,L
\\
\epsilon,&l&=&L+1 
\end{matrix}\right.
\end{align}
$$

$$
\begin{matrix}
\boldsymbol{Z}&&&&
\\
&\searrow&&&
\\
&&\boldsymbol{Y}&&
\\
&\nearrow&&\nwarrow&
\\
\boldsymbol{\Lambda}&&&&\boldsymbol{R}
\\
\end{matrix}
$$





## Mixture of Factor Analyzer (MFA)

## Overview

$\boldsymbol{Y}$: Observation

$\boldsymbol{Z}$: Latent variables (Factor analysis)

$\boldsymbol{S}$: Latent variables (Mixture)

$\boldsymbol{\Lambda}$: Factor loading 

$\boldsymbol{R}$: Uniqueness

$\boldsymbol{\pi}$: Mixture weights

## Model

$$
\begin{align}
p(\boldsymbol{Y},\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R},\boldsymbol{\pi})&=
p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R})p(\boldsymbol{Z})p(\boldsymbol{S}|\boldsymbol{\pi})p(\boldsymbol{\Lambda})p(\boldsymbol{R})p(\boldsymbol{\pi})
\\
p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R})
&=\prod_{t=1}^T\prod_{d=1}^D\prod_{k=1}^K\mathcal{N}(\boldsymbol{y}_{d,t}|\boldsymbol{\Lambda}_d^{(k)^T}\boldsymbol{z}_t,\boldsymbol{R}_d^{(k)^{−1}})^{s_t^{(k)}}
\\
p(\boldsymbol{S}|\boldsymbol{\pi})
&=\prod_{t=1}^{T}\mathrm{Cat}(s_t|\boldsymbol{\pi})
\\
p(\boldsymbol{\Lambda})&=\prod_{d=1}^{D}\prod_{k=1}^{K}\mathcal{N}(\boldsymbol{\Lambda}_d^{(k)}|\boldsymbol{\mu}^{(k)}_{\boldsymbol{\Lambda}_d},\boldsymbol{\Sigma}^{(k)}_{\boldsymbol{\Lambda}_d})
\\
p(\boldsymbol{R})
&=\prod_{d=1}^{D}\prod_{k=1}^{K}\mathrm{Gam}(\boldsymbol{R}_{d}^{(k)}|a_{R_d}^{(k)},b_{R_d}^{(k)})
\\
p(\boldsymbol{Z})&=\prod_t^T \mathcal{N}(z_{t}|\mu_{z},\Sigma_{z})=\prod_t^T \prod_{l=1}^{L + 1} \mathcal{N} (z_{l,t}|\mu_{z_l},\sigma_{z_l}^2)
\\
&\mu_{z_l}=\left\{\begin{matrix}
0,&l&=& 1,\cdots,L
\\
1,&l&=&L + 1
\end{matrix}\right.
\\
&\sigma_{z_l}^{2}=\left\{\begin{matrix}
1,&l&=&1,\cdots,L
\\
\epsilon,&l&=&L+1 
\end{matrix}\right.
\end{align}
$$
(R is diagonal)
$$
\begin{matrix}
\boldsymbol{Z}&&&&\boldsymbol{S}
\\
&\searrow&&\swarrow&
\\
&&\boldsymbol{Y}&&
\\
&\nearrow&&\nwarrow&
\\
\boldsymbol{\Lambda}&&&&\boldsymbol{R}
\\
\end{matrix}
$$


## Variational Inference

$$
\begin{align}
p(\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R},\boldsymbol{\pi}) \approx & q(\boldsymbol{Z},\boldsymbol{S})q(\boldsymbol{\Lambda})q(\boldsymbol{R})q(\boldsymbol{\pi})
\\
\ln p(\boldsymbol{Z},\boldsymbol{S}) =& \langle \ln p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R})\rangle_{q(\boldsymbol{\Lambda},\boldsymbol{R})}+ \ln p(\boldsymbol{Z})+\langle p(\boldsymbol{S}|\boldsymbol{\pi})\rangle_{q(\boldsymbol{\pi})} + C
\\
\ln q(\boldsymbol{\Lambda})=&\langle \ln p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R})\rangle_{q(\boldsymbol{Z},\boldsymbol{S}),q(\boldsymbol{R})}+\ln p(\boldsymbol{\Lambda})+C
\\
\ln q(\boldsymbol{R})=&\langle \ln p(\boldsymbol{Y}|\boldsymbol{Z},\boldsymbol{S},\boldsymbol{\Lambda},\boldsymbol{R})\rangle_{q(\boldsymbol{Z},\boldsymbol{S}),q(\boldsymbol{\Lambda})}+\ln q(\boldsymbol{R})+C
\\
\ln q(\boldsymbol{\pi})=&\langle\ln p(\boldsymbol{S}|\boldsymbol{\pi})\rangle_{q(\boldsymbol{S})}+\ln p(\boldsymbol{\pi}) +C
\end{align}
$$

