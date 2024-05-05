<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Document</title>
  </head>
  <body>
    <p id='env'></p>
	<?=  getenv('ES_HOST'); ?>
	<?=  $_SERVER['ES_HOST']; // don't work ?>
	<?=  $_ENV['ES_HOST']; ?>

	<?=  getenv('NEW_ENV'); ?>
	<?=  $_SERVER['NEW_ENV']; // don't work ?> 
	<?=  $_ENV['NEW_ENV']; ?>

	<?=  $_ENV['BASE_URL_SERVER']; ?>
	<?=  $_ENV['BASE_URL_PYTHON']; ?>
  </body>
</html>
