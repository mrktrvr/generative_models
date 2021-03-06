$$\begin{align}  "p(\\boldsymbol{S},\\boldsymbol{Z},\\boldsymbol{\\Phi},\\boldsymbol{\\Pi})=&p(\\boldsymbol{S}|\\boldsymbol{Z},\\boldsymbol{\\Pi})p(\\boldsymbol{Z}|\\boldsymbol{\\Phi})p(\\boldsymbol{\\Phi})p(\\boldsymbol{\\Pi})\n",
    "\\\\\n",
    "p(\\boldsymbol{S}|\\boldsymbol{Z},\\boldsymbol{\\Pi})&=\\prod_{b=1}^{B}\\prod_{t=1}^{T_b}\\prod_{c=1}^{C}\\mathrm{Cat}(\\boldsymbol{s}_{b,t}|\\boldsymbol{\\pi}_c)^{\\boldsymbol{z}_{b,t}^{(c)}}\n",
    "\\\\\n",
    "p(\\boldsymbol{Z}|\\boldsymbol{\\Phi})&=\\prod_{b=1}^{B}\\prod_{t=1}^{T_b}\\mathrm{Cat}(\\boldsymbol{z}_{b,t}|\\boldsymbol{\\phi}_b)\n",
    "\\\\\n",
    "p(\\boldsymbol{\\Phi})&=\\prod_{b=1}^{B}p(\\boldsymbol{\\phi}_b)=\\prod_{b=1}^{B}\\mathrm{Dir}(\\boldsymbol{\\phi}_b|\\boldsymbol{\\alpha}_{\\phi})\n",
    "\\\\\n",
    "p(\\boldsymbol{\\Pi})&=\\prod_{c=1}^{C}p(\\boldsymbol{\\pi}_c)=\\prod_{c=1}^{C}\\mathrm{Dir}(\\boldsymbol{\\pi}_c|\\boldsymbol{\\alpha}_{\\pi})\n",
\\end{align}$$
# Model
$$\alpha$$
