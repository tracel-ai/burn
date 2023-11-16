import Train from './train.ts?worker'

export function setupCounter(element: HTMLButtonElement) {
	let counter = 0
	const setCounter = (count: number) => {
		counter = count
		element.innerHTML = `count is ${counter}`
	}
	element.addEventListener('click', () => setCounter(counter + 1))
	setCounter(0)
}

export function setupTrain(
	element: HTMLButtonElement,
	fileElement: HTMLInputElement,
) {
	element.addEventListener('click', async () => {
		element.innerHTML = `Training...`
		element.disabled = true
		const train = new Train()
		const files = fileElement.files
		if (files != null) {
			const file = files[0]
			if (file != null) {
				let ab = await file.arrayBuffer()
				train.postMessage(ab, [ab])
				return
			}
		}
		train.postMessage('autotrain')
	})
}

export function setupAutotrain(
	element: HTMLInputElement,
	trainElement: HTMLButtonElement,
) {
	const autotrain = localStorage.getItem('autotrain')
	element.checked = autotrain != null
	if (element.checked) {
		trainElement.click()
	}
	element.addEventListener('click', () => {
		if (element.checked) {
			localStorage.setItem('autotrain', 'true')
		} else {
			localStorage.removeItem('autotrain')
		}
	})
}
