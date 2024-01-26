function fetchData() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('dataContainer').innerText = 'Data from Flask: ' + data.message;
        });
}
