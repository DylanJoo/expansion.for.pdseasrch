simplified:
	mkdir -p indexes
	python -m pyserini.index.lucene \
	    --collection JsonCollection \
	      --input data/simplified_corpus \
	      --index indexing/trec-pds-simplified/ \
	      --generator DefaultLuceneDocumentGenerator \
	      --threads 4

expanded:
	for folder in data/expanded_corpus/* ; do \
		file=$${folder##*/}; \
	        python -m pyserini.index.lucene \
	            --collection JsonCollection \
	              --input $${folder} \
	              --index indexing/trec-pds-expanded-$${file##*/}/ \
	              --generator DefaultLuceneDocumentGenerator \
	              --threads 4 ;\
	done
