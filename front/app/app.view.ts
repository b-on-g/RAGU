namespace $.$$ {

	type Request = {
		message: string
		files: string[]
	}

	type IndexRecord = {
		count: number
		names: string[]
	}

	export class $bog_RAGU_front_app extends $.$bog_RAGU_front_app {

		@ $mol_mem
		config_synced() {
			this.push_config()
			return true
		}

		@ $mol_mem
		override pages() {
			this.config_synced()
			return [
				this.Settings_page(),
				this.Documents(),
				this.Dialog(),
				... this.result() ? [ this.Result_page( this.version() ) ] : [],
			]
		}

		@ $mol_action
		override doc_files_add( next: readonly File[] ) {
			if( !next?.length ) return
			this.doc_files([ ... this.doc_files(), ... next ])
		}

		override doc_file_rows() {
			return this.doc_files().map( ( _, i ) => this.Doc_file( i ) )
		}

		override doc_file_name( index: number ) {
			return ( this.doc_files()[ index ] as File ).name
		}

		@ $mol_action
		override doc_file_remove( index: number ) {
			const files = [ ... this.doc_files() ]
			files.splice( index, 1 )
			this.doc_files( files )
		}

		override index_record_rows() {
			return this.index_records().map( ( _, i ) => this.Index_record( i ) )
		}

		override index_record_text( index: number ) {
			const rec = this.index_records()[ index ] as IndexRecord
			return `${ rec.count } doc(s): ${ rec.names.join( ', ' ) }`
		}

		@ $mol_mem
		override communication() {

			const history = this.history()
			if( history.length % 2 === 0 ) return

			const last = history[ history.length - 1 ] as Request

			try {
				const resp = $mol_fetch.json(
					this.api_url() + '/api/query',
					{
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ query: last.message }),
					},
				)
				this.history([ ... history, resp ])
			} catch( error: any ) {
				if( $mol_promise_like( error ) ) $mol_fail_hidden( error )
				if( $mol_fail_log( error ) ) {
					this.history([ ... history, { message: '\u{1F6D1}' + error.message, files: [] } ])
				}
			}

		}

		@ $mol_action
		override index_submit() {
			const text = this.doc_text()
			const files = this.doc_files() as File[]

			if( !text && !files.length ) return

			const form = new FormData()
			if( text ) form.append( 'text', text )
			for( const file of files ) {
				form.append( 'files', file )
			}

			const resp = $mol_fetch.json(
				this.api_url() + '/api/index',
				{
					method: 'POST',
					body: form,
				},
			) as { status: string; documents_count: number; names: string[]; total_documents: number }

			this.index_records([
				... this.index_records(),
				{ count: resp.documents_count, names: resp.names } as IndexRecord,
			])

			this.index_message( `Indexed ${ resp.documents_count } doc(s). Total: ${ resp.total_documents }` )
			this.doc_text( '' )
			this.doc_files( [] )
		}

		@ $mol_action
		push_config() {
			$mol_fetch.json(
				this.api_url() + '/api/config',
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ env: {
						LLM_API_KEY: this.llm_api_key(),
						LLM_BASE_URL: this.llm_base_url(),
						LLM_MODEL: this.llm_model(),
						LLM_RPM: this.llm_rpm(),
						EMBEDDER_API_KEY: this.embedder_api_key(),
						EMBEDDER_BASE_URL: this.embedder_base_url(),
						EMBEDDER_MODEL: this.embedder_model(),
						EMBEDDER_DIM: this.embedder_dim(),
					} }),
				},
			)
		}

		@ $mol_action
		override config_save() {
			this.push_config()
			this.config_message( 'Saved' )
		}

	}

}
