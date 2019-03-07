from distutils.core import setup
setup(
  name = 'truncated_normal',
  packages = ['truncated_normal'], # this must be the same as the name above
  version = '0.3',
  description = 'Package for running the TN test described in https://www.biorxiv.org/content/early/2018/11/05/463265',
  author = 'Jesse Min Zhang',
  author_email = 'jessez@stanford.edu',
  url = 'https://github.com/jessemzhang/tn_test',
  download_url = 'https://github.com/jessemzhang/tn_test/archive/0.3',
  keywords = ['truncated-normal', 'differential-expression', 'single-cell', 'rna-seq', 'clustering', 'tn-test'], 
  classifiers = [],
)
