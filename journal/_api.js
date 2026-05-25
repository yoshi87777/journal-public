// journal :: API base shim.
//
// Pages call http://localhost:5757 (the feedback server). This file just
// exposes window.lifeOsApi so pages can introspect, and is a no-op when
// served locally. The original life-os version routed traffic through a
// public tunnel — that's stripped out here for the local-only build.

(function () {
  var API_BASE = 'http://localhost:5757';
  window.lifeOsApi = {
    base: API_BASE,
    isLocal: true,
  };
})();
