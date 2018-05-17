# radvel-search
RV Planet Search Pipeline Based on RadVel

Use radvel setup files to load:
    - parameters for "known" planets
    - data and instruments
    - fix/vary within search
    - fitting (search) basis
    
    
`rvsearch periodogram -t [bic, aic, ls] -n 1 -s path-to-setup`

`rvsearch find -s path-to-setup`

`rvsearch inject -s path-to-setup`

`rvsearch plot -t peri -s path-to-setup`


Do we require users to set up their config files with 
enough planet parameters for the searches? Or any planet parameters at all?
Turns out building a posterior 

How do we pass the results of `rvsearach find` back into setup file to
calculate next periodogram, and iterate?

