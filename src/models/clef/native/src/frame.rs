//! Big-endian frames shared with `models.clef.ipc`.
//!
//! A request payload is `u8 opcode` plus the raw HTTP body. A response
//! payload is `u16 status` plus the JSON bytes forwarded on the socket.

use std::io::{Read, Write};

use anyhow::{Context, Result, bail};

pub(crate) const MAX_FRAME: usize = 16 * 1024 * 1024;
pub(crate) const OPCODE_DECIDE: u8 = 1;

pub(crate) struct WorkerResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn write_request(writer: &mut impl Write, opcode: u8, body: &[u8]) -> Result<()> {
    let length = 1 + body.len();
    if length > MAX_FRAME {
        bail!("request frame exceeds {MAX_FRAME} bytes");
    }
    writer
        .write_all(&(length as u32).to_be_bytes())
        .context("writing the executor frame")?;
    writer
        .write_all(&[opcode])
        .context("writing the executor frame")?;
    writer
        .write_all(body)
        .context("writing the executor frame")?;
    writer.flush().context("writing the executor frame")?;
    Ok(())
}

pub(crate) fn read_response(reader: &mut impl Read) -> Result<WorkerResponse> {
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .context("reading the executor frame")?;
    let length = u32::from_be_bytes(len_buf) as usize;
    if !(2..=MAX_FRAME).contains(&length) {
        bail!("executor frame length {length} is outside 2..={MAX_FRAME}");
    }
    let mut payload = vec![0u8; length];
    reader
        .read_exact(&mut payload)
        .context("reading the executor frame")?;
    let status = u16::from_be_bytes([payload[0], payload[1]]);
    Ok(WorkerResponse {
        status,
        body: payload[2..].to_vec(),
    })
}
