<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1.0" />
		<link rel="stylesheet" href="style.css" />
		<title>Machine Learning Project</title>
	</head>
	<body>
		<header class="header">
			<h1><a href="/">ml-project</a></h1>

			<nav class="nav">
				<a href="?v=search" class="button">Search</a>
				<a href="?v=add" class="button">Add</a>
			</nav>
		</header>

		<?php
			if (empty($_GET['v']) || ($_GET['v'] == 'search')) {
				include 'search.html';
			} else if ($_GET['v'] == 'add'){
				include 'add.html';
			} else if ($_GET['v'] === 'article' && !empty($_GET['id'])) {
				include 'article.html';
			} else {
				include '404.html';
			}
		?>

		<script type="text/javascript">
			document.addEventListener("DOMContentLoaded", () => {
				// <!-- console.log('DOM loaded with JavaScript'); -->

				// const searchBtn = document.getElementById("searchBtn");
				// searchBtn.addEventListener("click", () => {
				// 	window.location.href = "?v=search";
				// });

				// const addBtn = document.getElementById("addBtn");
				// addBtn.addEventListener("click", () => {
				// 	window.location.href = "?v=add";
				// });
			});
		</script>
	</body>
</html>