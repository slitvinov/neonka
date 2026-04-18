# neonka

```
$ md5sum data/train.raw data/train.lob.gz data/train.csv
8b9f8a864c77caa0e83523fa5a804b48  data/train.raw
cfadc1524cd19afda7a64b8f30405e32  data/train.lob.gz
0d92bf14fc90c1f89525d47bd31052e6  data/train.csv
```

    Think of the book as a 2-state queue for orders:                                                                         
    - State T = "at top" (best level)                                                                                        
    - State D = "deep" (levels 1–7)                                                                                          
    - 4 transitions:                                                                                                         
      - tp: new order born at T (rate a)                                                                                     
      - tm: T→gone (cancel or execute from top, rate μ·n_T)                                                                  
      - dp: new order born at D (rate b)                                                                                     
      - dm: D→gone (rate ν·n_D)                                                                                              
      - (plus T↔D shifts — when best moves, top rolls into deep or vice versa)