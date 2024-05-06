indexMapping = {
	'properties': {
		'id': {
			'type': 'text'
		},
		'url': {
			'type': 'text'
		},
		'title': {
			'type': 'text'
		},
		'text': {
			'type': 'text'
		},
		'title_embeddings': {
			'type': 'dense_vector',
			'dims': 768,
			'index': False,
		},
		'text_embeddings': {
			'type': 'dense_vector',
			'dims': 768,
			'index': True,
			'similarity': 'l2_norm'
		}
	}
}