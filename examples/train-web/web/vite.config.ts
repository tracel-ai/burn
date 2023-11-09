import { defineConfig } from 'vite'

export default defineConfig({
	plugins: [
		{
			// https://gist.github.com/mizchi/afcc5cf233c9e6943720fde4b4579a2b
			name: 'isolation',
			configureServer(server) {
				server.middlewares.use((_req, res, next) => {
					res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
					res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
					next()
				})
			},
		},
	],
	server: {
		fs: {
			// Allow serving files from one level up to the project root
			allow: ['..'],
		},
	},
})
