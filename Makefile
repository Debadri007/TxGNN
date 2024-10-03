# Development commands
copy-env:
	cp .env.example .env

pull:
	docker-compose pull --parallel

build:
	docker-compose build

up:
	docker-compose up -d

status:
	docker-compose ps

logs:
	docker-compose logs -tf webapp

down:
	docker-compose down --remove-orphans

restart:
	@$(MAKE) down; \
	$(MAKE) up

run:
	@$(MAKE) copy-env; \
	$(MAKE) pull; \
	$(MAKE) build; \
	$(MAKE) up; \
	$(MAKE) status; \
	$(MAKE) setup; \
	$(MAKE) restart; \
	$(MAKE) logs

purge:
	if [ ! -f .env ]; then \
		touch .env; \
	fi
	docker-compose down -v --rmi all --remove-orphans
	if [ -f .env ]; then \
		rm .env; \
	fi
