expanded:
	for folder in data/expanded_corpus/sim*12000 ; do \
		file=$${folder##*/}; \
	        python -m pyserini.index.lucene \
	            --collection JsonCollection \
	              --input $${folder} \
	              --index indexing/trec-pds-expanded-$${file##*/}/ \
	              --generator DefaultLuceneDocumentGenerator \
	              --threads 4 ;\
	done

