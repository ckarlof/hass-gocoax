"""Async client for goCoax MoCA adapters."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

import aiohttp
import yarl
from aiohttp import BasicAuth

from .exceptions import (
    GoCoaxAuthError,
    GoCoaxConnectionError,
    GoCoaxError,
    GoCoaxParseError,
    GoCoaxTimeoutError,
)
from .models import (
    AdapterStatus,
    EthernetPackets,
    NetworkPeer,
    PacketStats,
    PhyRate,
    SignalQuality,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession

LOG = logging.getLogger(__name__)

# API endpoints - JSON data
ENDPOINT_MAC = "/ms/1/0x103/GET"
ENDPOINT_LOCAL_INFO = "/ms/0/0x15"
ENDPOINT_FRAME_INFO = "/ms/0/0x14"
ENDPOINT_FMR_INFO = "/ms/0/0x1D"
ENDPOINT_NODE_INFO = "/ms/0/0x16"  # POST with node ID; firmware 2.0.16+ per-node query
ENDPOINT_PRIVACY = "/ms/0/0x1059/GET"  # firmware 2.0.16+ security mode mask (GET)
ENDPOINT_MPS = "/ms/0/0x18"  # MoCA Protected Setup
ENDPOINT_CONFIG = "/ms/0/0x1003/GET"  # firmware 2.0.16+ LOF endpoint (GET)

# HTML page endpoints
ENDPOINT_PHY_RATES = "/phyRates.html"
ENDPOINT_SECURITY_HTML = "/security.html"
ENDPOINT_STATUS_HTML = "/index.html"

# data indices from decoded format
LOCAL_INFO_NODE_ID_IDX = 3
LOCAL_INFO_NC_NODE_IDX = 4
LOCAL_INFO_LINK_STATUS_IDX = 5
LOCAL_INFO_MOCA_VER_IDX = 11
LOCAL_INFO_NODE_BITMASK_IDX = 12  # bitmask of active node IDs on network

NODE_INFO_PHY_RATE_IDX = 3  # data[3] bits 0-15 = PHY rate in Mbps; upper byte = MoCA version

LOCAL_INFO_FW_VERSION_IDX = 21  # ASCII bytes of SDK/firmware version start here

ENDPOINT_PHY_INFO = "/ms/0/0x7f"   # MxL371x only: data[2]=first channel, data[3]=num channels
ENDPOINT_CHIP_ID = "/ms/1/0x303/GET"  # chip ID: 0x15=MXL370x, 0x16=MXL371x

_CHIP_NAMES = {0x15: "MXL370x", 0x16: "MXL371x"}

FRAME_TX_GOOD_IDX = 12
FRAME_TX_BAD_IDX = 30
FRAME_TX_DROP_IDX = 48
FRAME_RX_GOOD_IDX = 66
FRAME_RX_BAD_IDX = 84
FRAME_RX_DROP_IDX = 102

DEFAULT_TIMEOUT = 15


class GoCoaxClient:
    """Async client for communicating with goCoax MoCA adapters."""

    def __init__(
        self,
        host: str,
        username: str = "admin",
        password: str = "gocoax",
        session: ClientSession | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the goCoax client."""
        self._host = host
        self._username = username
        self._password = password
        self._session = session
        self._owns_session = session is None
        self._timeout = timeout
        self._base_url = f"http://{host}"
        self._csrf_token: str | None = None  # None = not yet fetched

    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the client session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _fetch_csrf_token(self) -> None:
        """Fetch CSRF token from the device main page.

        Firmware 2.0.16.0+ requires an X-CSRF-TOKEN header on all POST
        requests. The token is set as a cookie on the first page load.
        """
        try:
            session = await self._get_session()
            url = f"{self._base_url}/index.html"
            auth = BasicAuth(self._username, self._password)
            async with session.get(url, auth=auth) as resp:
                if resp.status == 200:
                    token = resp.cookies.get("csrf_token")
                    if token:
                        self._csrf_token = token.value
                        LOG.debug("Obtained CSRF token from %s", self._host)
                        return
        except Exception as err:
            LOG.debug("Could not fetch CSRF token from %s: %s", self._host, err)
        # Older firmware has no CSRF protection — mark as fetched but absent
        self._csrf_token = ""

    async def _request(
        self, endpoint: str, method: str = "GET", post_data: str = ""
    ) -> dict | str:
        """Make an authenticated request to the adapter."""
        session = await self._get_session()
        url = f"{self._base_url}{endpoint}"
        auth = BasicAuth(self._username, self._password)

        try:
            if method == "POST":
                # CSRF fetch gets its own implicit timeout (aiohttp connector),
                # separate from the per-request timeout below.
                if self._csrf_token is None:
                    await self._fetch_csrf_token()

            async with asyncio.timeout(self._timeout):
                if method == "POST":
                    headers: dict[str, str] = {
                        "content-type": "application/x-www-form-urlencoded",
                        "Accept": "text/html, */*",
                    }

                    # Prefer the token currently in the session jar — intermediate
                    # GET responses may have rotated it since our last fetch.
                    host_url = yarl.URL(f"http://{self._host}/")
                    jar_token = session.cookie_jar.filter_cookies(host_url).get(
                        "csrf_token"
                    )
                    effective_token: str = (
                        jar_token.value if jar_token else (self._csrf_token or "")
                    )

                    # The firmware uses the Double Submit Cookie pattern:
                    # it validates that the X-CSRF-TOKEN header matches the
                    # csrf_token cookie.  Explicitly send the cookie so that
                    # this works even when the session jar is not replaying it
                    # (e.g. HA's shared session with IP-address cookie restrictions).
                    request_cookies: dict[str, str] = {}
                    if effective_token:
                        headers["X-CSRF-TOKEN"] = effective_token
                        request_cookies["csrf_token"] = effective_token

                    body = f'{{"data":[{post_data}]}}'
                    async with session.post(
                        url, data=body, auth=auth, headers=headers,
                        cookies=request_cookies,
                    ) as resp:
                        # 400 may mean CSRF token expired — refresh and retry once
                        if resp.status == 400 and effective_token:
                            self._csrf_token = None
                            await self._fetch_csrf_token()
                            jar_token = session.cookie_jar.filter_cookies(
                                host_url
                            ).get("csrf_token")
                            effective_token = (
                                jar_token.value
                                if jar_token
                                else (self._csrf_token or "")
                            )
                            # Rebuild header and cookie with fresh token
                            headers.pop("X-CSRF-TOKEN", None)
                            request_cookies.clear()
                            if effective_token:
                                headers["X-CSRF-TOKEN"] = effective_token
                                request_cookies["csrf_token"] = effective_token
                            async with session.post(
                                url, data=body, auth=auth, headers=headers,
                                cookies=request_cookies,
                            ) as resp2:
                                return await self._handle_response(resp2, url)
                        return await self._handle_response(resp, url)
                else:
                    async with session.get(url, auth=auth) as resp:
                        return await self._handle_response(resp, url)

        except TimeoutError as err:
            raise GoCoaxTimeoutError(
                f"Timeout connecting to goCoax adapter at {self._host}"
            ) from err
        except aiohttp.ClientError as err:
            raise GoCoaxConnectionError(
                f"Error connecting to goCoax adapter at {self._host}: {err}"
            ) from err

    async def _handle_response(
        self, resp: aiohttp.ClientResponse, url: str
    ) -> dict | str:
        """Handle HTTP response."""
        if resp.status == 401:
            raise GoCoaxAuthError(
                f"Authentication failed for goCoax adapter at {self._host}"
            )
        if resp.status != 200:
            raise GoCoaxConnectionError(
                f"HTTP {resp.status} from goCoax adapter: {url}"
            )

        content_type = resp.content_type or ""
        if "json" in content_type or url.endswith("/GET"):
            return await resp.json()
        return await resp.text()

    def _parse_hex_value(self, hex_str: str) -> int:
        """Parse hex string to integer."""
        try:
            return int(hex_str, 16)
        except (ValueError, TypeError):
            return 0

    def _parse_64bit_value(self, data: list[str], high_idx: int) -> int:
        """Parse 64-bit value from two consecutive hex strings."""
        if high_idx + 1 >= len(data):
            return 0
        high = self._parse_hex_value(data[high_idx]) & 0xFFFFFFFF
        low = self._parse_hex_value(data[high_idx + 1])
        return (high * 4294967296) + low

    def _hex_to_mac(self, hi: int, lo: int) -> str:
        """Convert two integers to MAC address string."""
        # extract bytes: hi contains first 4 bytes, lo contains last 2
        b1 = (hi >> 24) & 0xFF
        b2 = (hi >> 16) & 0xFF
        b3 = (hi >> 8) & 0xFF
        b4 = hi & 0xFF
        b5 = (lo >> 24) & 0xFF
        b6 = (lo >> 16) & 0xFF
        return f"{b1:02x}:{b2:02x}:{b3:02x}:{b4:02x}:{b5:02x}:{b6:02x}"

    def _hex_to_ascii_str(self, data: list[str], start_idx: int) -> str:
        """Decode a null-terminated ASCII string packed into 32-bit hex words."""
        result = []
        for i in range(start_idx, len(data)):
            word = self._parse_hex_value(data[i])
            for shift in (24, 16, 8, 0):
                byte = (word >> shift) & 0xFF
                if byte == 0:
                    return "".join(result)
                if 0x20 <= byte < 0x80:
                    result.append(chr(byte))
                else:
                    return "".join(result)
        return "".join(result)

    def _parse_moca_version(self, ver_int: int) -> str:
        """Parse MoCA version from integer value."""
        # version is encoded as major * 16 + minor
        if ver_int == 0x20:
            return "2.0"
        if ver_int == 0x25:
            return "2.5"
        if ver_int >= 0x20:
            major = ver_int >> 4
            minor = ver_int & 0x0F
            return f"{major}.{minor}"
        return "unknown"

    async def get_mac_address(self) -> str:
        """Get the MAC address of the adapter."""
        try:
            resp = await self._request(ENDPOINT_MAC)
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                if len(data) >= 2:
                    hi = self._parse_hex_value(data[0])
                    lo = self._parse_hex_value(data[1])
                    return self._hex_to_mac(hi, lo)
        except GoCoaxParseError:
            LOG.debug("Failed to parse MAC address")
        return ""

    async def get_local_info(self) -> dict:
        """Get local adapter info (link status, MoCA version, etc.)."""
        # Empty data body {"data":[]} is correct for this endpoint — verified
        # against firmware 2.0.16.0 device (devStatus.html form value="").
        resp = await self._request(ENDPOINT_LOCAL_INFO, method="POST")
        if isinstance(resp, dict) and "data" in resp:
            data = resp["data"]
            LOG.debug("Local info response (raw): %s", data)
            fw_version = (
                self._hex_to_ascii_str(data, LOCAL_INFO_FW_VERSION_IDX)
                if len(data) > LOCAL_INFO_FW_VERSION_IDX
                else None
            )
            return {
                "link_status": self._parse_hex_value(data[LOCAL_INFO_LINK_STATUS_IDX])
                if len(data) > LOCAL_INFO_LINK_STATUS_IDX
                else 0,
                "moca_version": self._parse_hex_value(data[LOCAL_INFO_MOCA_VER_IDX])
                if len(data) > LOCAL_INFO_MOCA_VER_IDX
                else 0,
                "node_id": self._parse_hex_value(data[LOCAL_INFO_NODE_ID_IDX])
                if len(data) > LOCAL_INFO_NODE_ID_IDX
                else 0,
                "nc_node_id": self._parse_hex_value(data[LOCAL_INFO_NC_NODE_IDX])
                if len(data) > LOCAL_INFO_NC_NODE_IDX
                else 0,
                "node_bitmask": self._parse_hex_value(data[LOCAL_INFO_NODE_BITMASK_IDX])
                if len(data) > LOCAL_INFO_NODE_BITMASK_IDX
                else 0,
                "firmware_version": fw_version or None,
                "raw_data": data,
            }
        raise GoCoaxParseError("Invalid local info response")

    async def get_frame_info(self) -> EthernetPackets:
        """Get ethernet tx/rx packet statistics."""
        resp = await self._request(ENDPOINT_FRAME_INFO, method="POST", post_data="0")
        if isinstance(resp, dict) and "data" in resp:
            data = resp["data"]
            tx_good = self._parse_64bit_value(data, FRAME_TX_GOOD_IDX)
            tx_bad = self._parse_64bit_value(data, FRAME_TX_BAD_IDX)
            tx_drop = self._parse_64bit_value(data, FRAME_TX_DROP_IDX)
            rx_good = self._parse_64bit_value(data, FRAME_RX_GOOD_IDX)
            rx_bad = self._parse_64bit_value(data, FRAME_RX_BAD_IDX)
            rx_drop = self._parse_64bit_value(data, FRAME_RX_DROP_IDX)

            return EthernetPackets(
                tx=PacketStats(ok=tx_good, bad=tx_bad, dropped=tx_drop),
                rx=PacketStats(ok=rx_good, bad=rx_bad, dropped=rx_drop),
            )
        raise GoCoaxParseError("Invalid frame info response")

    async def get_node_info(
        self, local_node_id: int = -1, node_bitmask: int = 0
    ) -> list[NetworkPeer]:
        """Get information about peer nodes on the MoCA network.

        Firmware 2.0.16+ removed the all-nodes endpoint. Instead, the local
        info response includes a node_bitmask indicating which node IDs are
        active; each active node is then queried individually via the net info
        endpoint with its node ID in the request body.

        Response layout per node: data[0]=mac_hi, data[1]=mac_lo, data[4]=moca_ver
        """
        peers: list[NetworkPeer] = []
        if node_bitmask == 0:
            return peers
        try:
            for node_id in range(16):
                if not (node_bitmask & (1 << node_id)):
                    continue
                if node_id == local_node_id:
                    continue
                resp = await self._request(
                    ENDPOINT_NODE_INFO, method="POST", post_data=str(node_id)
                )
                if not (isinstance(resp, dict) and "data" in resp):
                    continue
                data = resp["data"]
                LOG.debug("Node info for node %d (raw): %s", node_id, data)
                if len(data) < 5:
                    continue
                mac_hi = self._parse_hex_value(data[0])
                mac_lo = self._parse_hex_value(data[1])
                mac_addr = self._hex_to_mac(mac_hi, mac_lo)
                if mac_addr == "00:00:00:00:00:00":
                    continue
                moca_ver_int = self._parse_hex_value(data[4])
                moca_ver = self._parse_moca_version(moca_ver_int)
                # data[3] encodes: upper byte = MoCA version, lower 16 bits = PHY rate (Mbps)
                rate_raw = self._parse_hex_value(data[NODE_INFO_PHY_RATE_IDX])
                phy_rate = rate_raw & 0xFFFF  # MoCA links are symmetric: TX == RX
                peers.append(
                    NetworkPeer(
                        node_id=node_id,
                        mac_address=mac_addr,
                        moca_version=moca_ver,
                        tx_phy_rate=phy_rate,
                        rx_phy_rate=phy_rate,
                    )
                )
                LOG.debug(
                    "Parsed peer: node_id=%d, mac=%s, version=%s",
                    node_id,
                    mac_addr,
                    moca_ver,
                )
        except GoCoaxError as err:
            LOG.debug("Failed to get node info: %s", err)
        return peers

    async def get_phy_rates(self) -> list[PhyRate]:
        """Parse PHY rates from the phyRates.html page."""
        rates: list[PhyRate] = []
        try:
            html = await self._request(ENDPOINT_PHY_RATES)
            if isinstance(html, str):
                rates = self._parse_phy_rates_html(html)
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug(f"Failed to get PHY rates: {err}")
        return rates

    def _parse_phy_rates_html(self, html: str) -> list[PhyRate]:
        """Parse PHY rates table from HTML."""
        rates: list[PhyRate] = []
        # look for table with PHY rates
        # format varies by firmware, typical pattern:
        # MAC addr | TX rate | RX rate
        table_match = re.search(
            r"<table[^>]*>(.*?)</table>", html, re.IGNORECASE | re.DOTALL
        )
        if not table_match:
            return rates

        table_html = table_match.group(1)
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.IGNORECASE | re.DOTALL)

        for row in rows[1:]:  # skip header row
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
            if len(cells) >= 3:
                mac_match = re.search(r"([0-9a-fA-F:]{17})", cells[0].strip())
                if mac_match:
                    mac = mac_match.group(1).lower()
                    try:
                        tx_rate = int(re.sub(r"[^\d]", "", cells[1]) or "0")
                        rx_rate = int(re.sub(r"[^\d]", "", cells[2]) or "0")
                        rates.append(
                            PhyRate(
                                source_mac="",  # filled by caller
                                target_mac=mac,
                                tx_rate=tx_rate,
                                rx_rate=rx_rate,
                            )
                        )
                    except ValueError:
                        continue
        return rates

    async def get_phy_info(self) -> dict:
        """Get PHY info including channel count (MXL371x only).

        /ms/0/0x7f: data[2]=first channel (MHz), data[3]=number of bonded channels.
        """
        result: dict = {"channel_count": None, "first_channel": None}
        try:
            resp = await self._request(ENDPOINT_PHY_INFO, method="POST")
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                LOG.debug("PHY info response (raw): %s", data)
                if len(data) >= 4:
                    first_ch = self._parse_hex_value(data[2])
                    num_ch = self._parse_hex_value(data[3])
                    if num_ch > 0:
                        result["channel_count"] = num_ch
                    if first_ch > 0:
                        result["first_channel"] = first_ch
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get PHY info: %s", err)
        return result

    async def get_chip_id(self) -> str | None:
        """Get chip name from chip ID register.

        /ms/1/0x303/GET: returns chip ID; 0x15=MXL370x, 0x16=MXL371x.
        """
        try:
            resp = await self._request(ENDPOINT_CHIP_ID)
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                chip_id = self._parse_hex_value(data[0])
                return _CHIP_NAMES.get(chip_id, f"Unknown(0x{chip_id:02x})")
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get chip ID: %s", err)
        return None

    async def get_privacy_info(self) -> dict:
        """Get MoCA privacy/encryption settings.

        Returns dict with encryption_enabled.
        Endpoint /ms/0/0x1059/GET returns security mode mask; non-zero = enabled.
        """
        result: dict = {"encryption_enabled": None}
        try:
            resp = await self._request(ENDPOINT_PRIVACY)
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                LOG.debug("Privacy info response (raw): %s", data)
                if len(data) >= 1:
                    privacy_val = self._parse_hex_value(data[0])
                    result["encryption_enabled"] = privacy_val != 0
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get privacy info: %s", err)
        return result

    async def get_fmr_info(self) -> dict:
        """Get FMR (Frequency/Modulation/Rate) information.

        May contain signal quality data like SNR, power levels.
        """
        result: dict = {
            "snr": None,
            "tx_power": None,
            "rx_power": None,
            "bit_loading": None,
        }
        try:
            resp = await self._request(ENDPOINT_FMR_INFO, method="POST", post_data="")
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                LOG.debug("FMR info response (raw): %s", data)
                # format varies; log for analysis and attempt parsing
                # commonly: SNR, power levels, modulation info
                # will need real device data to finalize parsing
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get FMR info: %s", err)
        return result

    async def get_config_info(self) -> dict:
        """Get configuration parameters including frequency band.

        Returns LOF (lowest operating frequency) in MHz and band name.
        Endpoint /ms/0/0x1003/GET returns current LOF value.
        """
        result: dict = {
            "frequency_band": None,
            "lof": None,
            "channel_count": None,
        }
        try:
            resp = await self._request(ENDPOINT_CONFIG)
            if isinstance(resp, dict) and "data" in resp:
                data = resp["data"]
                LOG.debug("Config info response (raw): %s", data)
                if len(data) >= 1:
                    lof = self._parse_hex_value(data[0])
                    if 1000 <= lof <= 1700:  # sanity check for MHz range
                        result["lof"] = lof
                        result["frequency_band"] = self._lof_to_band(lof)
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get config info: %s", err)
        return result

    def _lof_to_band(self, lof: int) -> str:
        """Convert LOF (lowest operating frequency) to band name."""
        # MoCA 2.5 band definitions (approximate)
        if lof >= 1400:
            return "D-High"
        if lof >= 1225:
            return "D-Mid"
        if lof >= 1125:
            return "Extended-D"
        if lof >= 1000:
            return "D-Low"
        return f"Unknown ({lof} MHz)"

    async def get_status_page(self) -> dict:
        """Parse main status HTML page for additional info."""
        result: dict = {}
        try:
            html = await self._request(ENDPOINT_STATUS_HTML)
            if isinstance(html, str):
                LOG.debug("Status page HTML (first 2000 chars): %s", html[:2000])

                # firmware version — plain text or JS variable
                fw_match = re.search(
                    r"(?:firmware[_\s]*(?:version)?[:\s\"'=]+|[\"'](?:fw|firmware)[_]?(?:ver|version)?[\"']\s*[:=]\s*[\"'])([0-9][0-9.]+)",
                    html,
                    re.IGNORECASE,
                )
                if fw_match:
                    result["firmware_version"] = fw_match.group(1)

                # model name — plain text or JS variable
                model_match = re.search(
                    r"(?:model[:\s\"'=]+|[\"']model[\"']\s*[:=]\s*[\"'])(MA\d+\w*|WF-\d+\w*|FCA\d+)",
                    html,
                    re.IGNORECASE,
                )
                if model_match:
                    result["model"] = model_match.group(1).upper()

                # channel count
                channel_match = re.search(
                    r"(\d+)\s*channels?",
                    html,
                    re.IGNORECASE,
                )
                if channel_match:
                    result["channel_count"] = int(channel_match.group(1))

                LOG.debug("Status page parsed values: %s", result)
        except (GoCoaxConnectionError, GoCoaxParseError) as err:
            LOG.debug("Failed to get status page: %s", err)
        return result

    async def get_status(self) -> AdapterStatus:
        """Get complete adapter status."""
        mac_address = await self.get_mac_address()
        local_info = await self.get_local_info()

        link_status = local_info.get("link_status", 0) == 1
        moca_ver_int = local_info.get("moca_version", 0)
        moca_version = self._parse_moca_version(moca_ver_int)
        node_id = local_info.get("node_id", 0)
        nc_node_id = local_info.get("nc_node_id", 0)
        node_bitmask = local_info.get("node_bitmask", 0)
        is_nc = node_id == nc_node_id

        packets = await self.get_frame_info()
        peers = await self.get_node_info(local_node_id=node_id, node_bitmask=node_bitmask)

        # Build PHY rate list from per-node API data (HTML scraping is unreliable on
        # firmware 2.0.16.0 because phyRates.html renders its table via JavaScript).
        phy_rates = [
            PhyRate(
                source_mac=mac_address,
                target_mac=peer.mac_address,
                tx_rate=peer.tx_phy_rate,
                rx_rate=peer.rx_phy_rate,
            )
            for peer in peers
            if peer.tx_phy_rate > 0
        ]

        # fetch additional data for enhanced monitoring
        privacy_info = await self.get_privacy_info()
        config_info = await self.get_config_info()
        phy_info = await self.get_phy_info()
        chip_name = await self.get_chip_id()
        status_page = await self.get_status_page()

        # fill in source mac for phy rates
        for rate in phy_rates:
            rate.source_mac = mac_address

        signal_quality = SignalQuality(
            snr=None,
            tx_power=None,
            rx_power=None,
            bit_loading=None,
        )

        # channel count: prefer dedicated PHY info endpoint, fall back to status page
        channel_count = (
            phy_info.get("channel_count")
            or config_info.get("channel_count")
            or status_page.get("channel_count")
        )

        # firmware version: prefer local info ASCII field, fall back to status page
        firmware_version = (
            local_info.get("firmware_version")
            or status_page.get("firmware_version")
        )

        # model: chip name from chip ID register, fall back to status page
        model = chip_name or status_page.get("model")

        return AdapterStatus(
            mac_address=mac_address,
            ip_address=self._host,
            moca_version=moca_version,
            link_status=link_status,
            packets=packets,
            network_peers=peers,
            phy_rates=phy_rates,
            node_id=node_id,
            nc_node_id=nc_node_id,
            network_controller=is_nc,
            firmware_version=firmware_version,
            model=model,
            frequency_band=config_info.get("frequency_band"),
            lof=config_info.get("lof"),
            encryption_enabled=privacy_info.get("encryption_enabled"),
            signal_quality=signal_quality,
            channel_count=channel_count,
        )

    async def test_connection(self) -> bool:
        """Test connection to the adapter."""
        try:
            mac = await self.get_mac_address()
            return bool(mac)
        except GoCoaxAuthError:
            raise
        except GoCoaxConnectionError:
            raise
        except Exception as err:
            LOG.debug(f"Connection test failed: {err}")
            return False
