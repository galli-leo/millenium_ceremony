#!/bin/bash


curl 'https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&category=0&problem=1571&mosmsg=Submission+received+with+ID+23155753' \
-XPOST \
-F 'problemid=1571' \
-F 'category=0' \
-F 'language=5' \
-F 'code=' \
-F 'codeupl=@submit4.cpp' \
-H 'Cookie: dd0fd2e506f96b096ef58f7edf831085=fb82b577776ad56459a88d507f4082bb; 90f64d63bc84d18534dd9db4561f7819=822db64d31874e2dedf1ac70fdc67aab8c944a9b6269267c59276ae77ee315af1039337; __utma=152284925.643624059.1554202707.1554202707.1554202707.1; __utmb=152284925.500.10.1554202707; __utmc=152284925; __utmt=1; __utmz=152284925.1554202707.1.1.utmcsr; dd0fd2e506f96b096ef58f7edf831085=38d0bbb6462c5d792586dcd362e50ecd; __utma=152284925.643624059.1554202707.1554202707.1554202707.1; __utmb=152284925.500.10.1554202707; __utmc=152284925; 90f64d63bc84d18534dd9db4561f7819=822db64d31874e2dedf1ac70fdc67aab8c944a9b6269267c59276ae77ee315af1039337' \
-H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
-H 'Origin: https://uva.onlinejudge.org' \
-H 'Accept-Encoding: br, gzip, deflate' \
-H 'Host: uva.onlinejudge.org' \
-H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.3 Safari/605.1.15' \
-H 'Accept-Language: en-us' \
-H 'Referer: https://uva.onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=submit_problem&problemid=1571&category=0' \
-H 'Connection: keep-alive'