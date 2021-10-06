# nlp-experiments

To get set up, 
 1. Clone this repo 
 2. `pip install -r requirements.txt`
 3. \[Optional\] Clone https://github.com/whoiswillma/fewnerdparse, https://github.com/whoiswillma/wifineparse and set them up as well. With all the repos, your directory structure might look something like:
    
    ```
    fewnerdparse
    fewnerdparse/inter
    fewnerdparse/intra
    fewnerdparse/supervised
    nlp-experiments
    wifineparse
    wifineparse/Documents
    wifineparse/FineNE
    ```
    
Style stuff:
  - 80 columns is a soft guide.
  - Typically a good idea to add type annotations, especially for methods which take in common data structures such as `dict`, `list`, etc. 
      - Less necessary for tokenizers and models because those types are often clear from the file you're working in.
  - Tests for "pure" methods are nice.
  - Even though there is close collaboration, it's still a good idea to branch and PR because
      - It let's you document the changes you made e.g. for weekly reports
      - Batch all your changes into one place for others to see
  - `util.py` contains code for loading indicators, logging, and checkpointing
