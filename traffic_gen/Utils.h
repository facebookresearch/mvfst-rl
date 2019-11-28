/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <fizz/crypto/test/TestUtil.h>
#include <fizz/protocol/CertificateVerifier.h>
#include <fizz/protocol/clock/test/Mocks.h>
#include <fizz/server/FizzServerContext.h>
#include <quic/handshake/QuicFizzFactory.h>

namespace quic {
namespace traffic_gen {

class DummyCertificateVerifier : public fizz::CertificateVerifier {
 public:
  ~DummyCertificateVerifier() override = default;

  void verify(const std::vector<std::shared_ptr<const fizz::PeerCert>>&)
      const override {
    return;
  }

  std::vector<fizz::Extension> getCertificateRequestExtensions()
      const override {
    return std::vector<fizz::Extension>();
  }
};

std::shared_ptr<fizz::SelfCert> testCert() {
  auto certificate = fizz::test::getCert(fizz::test::kP256Certificate);
  auto privKey = fizz::test::getPrivateKey(fizz::test::kP256Key);
  std::vector<folly::ssl::X509UniquePtr> certs;
  certs.emplace_back(std::move(certificate));
  return std::make_shared<fizz::SelfCertImpl<fizz::KeyType::P256>>(
      std::move(privKey), std::move(certs));
}

std::shared_ptr<fizz::server::FizzServerContext> createTestServerCtx() {
  auto cert = testCert();
  auto certManager = std::make_unique<fizz::server::CertManager>();
  certManager->addCert(std::move(cert), true);
  auto serverCtx = std::make_shared<fizz::server::FizzServerContext>();
  serverCtx->setFactory(std::make_shared<QuicFizzFactory>());
  serverCtx->setCertManager(std::move(certManager));
  serverCtx->setOmitEarlyRecordLayer(true);
  serverCtx->setClock(std::make_shared<fizz::test::MockClock>());
  return serverCtx;
}

}  // namespace traffic_gen
}  // namespace quic
