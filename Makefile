.PHONY: dev api web test wheels web-build docker-image docker-offline docker-run check-parity

api:
	.venv/bin/python -m uvicorn navedenie.app:app --reload --port 8091 --host 127.0.0.1

web:
	cd web && npm run dev

test:
	.venv/bin/python -m pytest -q

dev:
	bash -c 'trap "kill 0" EXIT; .venv/bin/python -m uvicorn navedenie.app:app --reload --port 8091 --host 127.0.0.1 & cd web && npm run dev'

# локальные wheel для офлайн-сборки образа (linux/amd64, python 3.12)
wheels:
	rm -rf wheels && .venv/bin/pip download -q fastapi "uvicorn[standard]" numpy "pydantic>=2" websockets \
		--platform manylinux2014_x86_64 --implementation cp --python-version 3.12 --only-binary=:all: -d wheels/

web-build:
	cd web && npm run build

# hermetic-сборка (npm ci из интернета на этапе сборки);
# GIT_SHA попадает в шапку приложения — видно, из какого коммита собран образ
docker-image:
	docker build --platform linux/amd64 --build-arg GIT_SHA=$$(git rev-parse --short HEAD) -t muholet:latest .

# полностью офлайн: сначала make web-build wheels, потом этот таргет
docker-offline:
	docker build -f Dockerfile.offline --platform linux/amd64 -t muholet:latest .

docker-run:
	docker run --rm -p 8080:8080 -v "$(PWD)/data:/app/data" muholet:latest

# сверка фолбэка localSim (JS) с эталоном python по траекториям цели
check-parity:
	.venv/bin/python tools/export_target_traj.py > /tmp/muholet-ref.json
	npx esbuild web/src/localSim.ts --bundle --format=esm --outfile=/tmp/muholet-ls.mjs --log-level=error
	node tools/check_local_sim.mjs /tmp/muholet-ref.json /tmp/muholet-ls.mjs
