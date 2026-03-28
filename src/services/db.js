import { Low } from 'lowdb';
import { JSONFile } from 'lowdb/node';
import { join } from 'path';

const adapter = new JSONFile(join(process.cwd(), 'data/db.json'));
export const db = new Low(adapter, { queue: [], users: {} });
await db.read();
