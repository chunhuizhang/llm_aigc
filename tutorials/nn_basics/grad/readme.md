
- 标量关于谁求导，返回的 shape，就跟谁 shape 保持一致；
    - $y=x^TAx$
        - $\nabla_xy=(A+A^T)x$
        - $\nabla_Ay=xx^T$