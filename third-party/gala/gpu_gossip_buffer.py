# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Gossip Buffer

:author: Mido Assran
:description: Class defines a shared-memory Gossip-Buffer, which allows
    multi-processed asynchronous agents on the same machine to communicate
    tensors to on one-another
"""

import copy
import torch
import logging

logging.basicConfig(level=logging.INFO)


class GossipBuffer:
    def __init__(
        self,
        topology,
        model,
        buffer_locks,
        read_events,
        write_events,
        sync_list,
        sync_freq=0,
    ):
        """ GossipBuffer """

        self.topology = topology
        self.num_learners = len(topology)
        self.sync_list = sync_list
        self.sync_freq = sync_freq

        # Initialize message buffer (4-item object):
        # [0] -> Msg-Tensor
        # [1] -> Events recording peers that have read the message
        # [2] -> Events recording peer that has written the message
        # [3] -> Lock for safe access of Msg-Tensor
        self.msg_buffer = []
        for rank in range(self.num_learners):
            msg = copy.deepcopy(model)
            msg.share_memory()
            r_events = read_events[rank]
            w_events = write_events[rank]
            lock = buffer_locks[rank]
            self.msg_buffer.append([msg, r_events, w_events, lock])

        # Initialize each Read-Buffer as 'read'
        for msg_buffer in self.msg_buffer:
            read_event_list = msg_buffer[1]
            for event in read_event_list:
                event.set()

    def write_message(self, rank, model, rotate=False):
        """
        Write agent 'rank's 'model' to a local 'boradcast buffer' that will be
        read by the out-neighbours defined in 'self.topology'.

        :param rank: Agent's rank in multi-agent graph topology
        :param model: Agent's torch neural network model
        :param rotate: Whether to alternate peers in graph topology

        Agents should only write to their own broadcast buffer:
            i.e., ensure 'model' belongs to agent 'rank'
        WARNING: setting rotate=True with sync_freq > 1 not currently supported
        """
        with torch.no_grad():

            # Get local broadcast-buffer
            msg_buffer = self.msg_buffer[rank]
            broadcast_buffer = msg_buffer[0]
            read_event_list = msg_buffer[1]
            write_event_list = msg_buffer[2]
            lock = msg_buffer[3]

            # Check if out-peers finished reading our last message
            out_peers, _ = self.topology[rank].get_peers()
            read_complete = True
            for peer in out_peers:
                if not read_event_list[peer].is_set():
                    read_complete = False
                    break

            # If peers done reading our last message, wait and clear events
            if read_complete:
                for peer in out_peers:
                    read_event = read_event_list[peer]
                    read_event.wait()
                    read_event.clear()
            # If not done reading, cannot write another message right now
            else:
                return

            # Update broadcast-buffer with new message
            # -- flatten params and multiply by mixing-weight
            num_peers = self.topology[rank].peers_per_itr
            with lock:
                for bp, p in zip(broadcast_buffer.parameters(), model.parameters()):
                    bp.data.copy_(p)
                    bp.data.div_(num_peers + 1)
                # -- mark message as 'written'
                out_peers, _ = self.topology[rank].get_peers(rotate)
                torch.cuda.current_stream().synchronize()
                for peer in out_peers:
                    write_event_list[peer].set()

    def aggregate_message(self, rank, model):
        """
        Average messages with local model:
        Average all in-neighbours' (defined in 'self.topology') parameters with
        agent 'rank's 'model' and copy the result into 'model'.

        Agents should only aggregate messages into their own model:
            i.e., ensure 'model belongs to agent 'rank'
        """
        with torch.no_grad():

            # Check if in-peers finished writing messages to broadcast buffers
            _, in_peers = self.topology[rank].get_peers()
            write_complete = True
            for peer in in_peers:
                peer_buffer = self.msg_buffer[peer]
                write_event = peer_buffer[2][rank]
                if not write_event.is_set():
                    write_complete = False
                    break

            # Check if any messages are excessively stale
            stale_assert = self.sync_list[rank] >= self.sync_freq

            # If peers done writing or message too stale, wait and clear events
            if write_complete or stale_assert:
                for peer in in_peers:
                    peer_buffer = self.msg_buffer[peer]
                    write_event = peer_buffer[2][rank]
                    write_event.wait()
                    write_event.clear()
                    self.sync_list[rank] = 0
            # Not done writing, but staleness is still tolerable
            else:
                self.sync_list[rank] += 1
                logging.info(
                    "GALA agent %s: staleness %s" % (rank, self.sync_list[rank])
                )
                return

            # Lazy-mixing of local params
            num_peers = self.topology[rank].peers_per_itr
            for p in model.parameters():
                p.data.div_(num_peers + 1)

            # Aggregate received messages
            for peer in in_peers:
                peer_buffer = self.msg_buffer[peer]
                lock = peer_buffer[3]
                with lock:
                    # Read message and update 'params'
                    peer_msg = peer_buffer[0]
                    for p, bp in zip(model.parameters(), peer_msg.parameters()):
                        p.data.add_(bp.to(p.device, non_blocking=True))
                    torch.cuda.current_stream().synchronize()
                    # Mark message as 'read'
                    peer_buffer[1][rank].set()
